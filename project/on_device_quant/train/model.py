"""
EdgeTransformer 模型定义

核心架构：混合注意力（Hybrid Attention）
  - CSA: Cross-Stream Attention，线性注意力，decay=0.99，关注跨流信息
  - HCA: Hybrid-Channel Attention，线性注意力，decay=0.999，长程场景覆盖
  - SWA: Sliding Window Attention，标准 softmax 因果注意力 + RoPE，保局部精度

三个分支通过可学习门控（softmax gate）加权融合。

多模态扩展（LLaVA 式前缀注入）：
  - VisionEncoder: 冻结 SigLIP ViT → MLP 投影 → 图像 token 前缀
  - EdgeTransformer.forward(x, images) 支持可选图像输入
  - 纯文本模式（images=None）与旧 checkpoint 完全兼容
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── 归一化层 ──

class RMSNorm(nn.Module):
    """RMSNorm：只做 RMS 归一化，省去 LayerNorm 的 mean 减法和 bias 参数

    相比 LayerNorm 减少 ~20% 计算：去掉 mean(centering) 和 bias(shift)，
    只保留 scale。权重初始化为 ones（恒等映射），训练时从全 1 开始学习缩放。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x.pow(2) 在 FP16 下可能溢出（>65504），所以强制在 FP32 中计算再转回
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def get_norm(dim, norm_type="rms", eps=1e-6):
    """归一化层工厂：norm_type='rms' → RMSNorm，'layernorm' → nn.LayerNorm"""
    if norm_type == "rms":
        return RMSNorm(dim, eps)
    return nn.LayerNorm(dim, eps=eps)


# ── 旋转位置编码（RoPE）──

class RoPE(nn.Module):
    """旋转位置编码：通过复数乘法实现相对位置编码"""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        # 频率倒数：1 / (base^(2i/dim))，i=0,1,...,dim/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, offset=0):
        """
        计算旋转角度嵌入
        x: [B, H, T, D] → 返回 [B, H, T, D]
        """
        t = torch.arange(offset, offset + x.shape[-2], device=x.device).type_as(self.inv_freq)
        freqs = t.unsqueeze(-1) @ self.inv_freq.unsqueeze(0)  # [T, D/2]
        return torch.cat([freqs, freqs], -1)  # [T, D]，两半相同


def apply_rope(x, emb):
    """
    应用旋转位置编码到 tensor
    用实数运算替代复数乘法（兼容 FP16，避免 ComplexHalf 实验性警告）
    x: [..., D] → [..., D]
    """
    s = x.shape[-1] // 2
    x1, x2 = x[..., :s], x[..., s:]
    e = emb[..., :s]  # emb = [θ, θ], 两半相同
    # 复数乘法 (x1+ix2)(θ+iθ) = θ(x1-x2) + i·θ(x1+x2)
    return torch.cat([e * (x1 - x2), e * (x1 + x2)], -1)


# ── 线性注意力（O(1) KV cache）──

def linear_attn(q, k, v, decay=0.99, eps=1e-6):
    """
    线性注意力：不存 KV cache，用累积状态矩阵 S 代替
    复杂度 O(T × D²)，内存恒定 O(D²)

    特征映射：relu(x)² + 0.1（替代 elu，避免 DirectML 兼容问题）

    q, k, v: [B, H, T, D]
    decay: 指数衰减因子，越小越关注近期
    注意：整个计算在 FP32 中进行（FP16 下 relu(x)² 容易溢出）
    """
    B, H, T, D = q.shape
    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    # 特征映射：relu(x)² 确保非负（类似 softmax 的正值特性），+0.1 防止全零
    qf = F.relu(q).pow(2) + 0.1
    kf = F.relu(k).pow(2) + 0.1
    vf = F.relu(v).pow(2) + 0.1
    # 累积状态矩阵 S: [B, H, D, D]，常数内存 O(D²)，不随序列长度增长
    # 归一化向量 z: [B, H, D]，跟踪 key 的累积和用于除法归一化
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    z = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)
    outs = []
    # 逐 token 循环：S 是递归状态，step i 依赖 step i-1 的 S，无法并行
    for i in range(T):
        ki = kf[:, :, i : i + 1]  # [B, H, 1, D]
        vi = vf[:, :, i : i + 1]
        qi = qf[:, :, i : i + 1]
        # 衰减旧记忆 + 写入新信息：S = decay·S + k^T @ v（外积更新）
        S = decay * S + ki.transpose(-1, -2) @ vi
        z = decay * z + ki.squeeze(2)
        # 查询累积状态：out = q @ S / (q @ z)，分母防止除零
        out = (qi @ S) / (qi @ z.unsqueeze(-1)).clamp(min=eps)
        outs.append(out)
    return torch.cat(outs, dim=-2).to(dtype)


# ── 三分支注意力模块 ──

class CSA(nn.Module):
    """
    Cross-Stream Attention（跨流注意力）
    线性注意力，1 个头，衰减因子 0.99
    功能：捕捉中等粒度的跨流信息，压缩比 4:1
    """

    def __init__(self, dim, hd):
        super().__init__()
        self.q = nn.Linear(dim, hd, bias=False)
        self.k = nn.Linear(dim, hd, bias=False)
        self.v = nn.Linear(dim, hd, bias=False)
        self.o = nn.Linear(hd, dim, bias=False)
        self.decay = 0.99

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, 1, -1).transpose(1, 2)
        k = self.k(x).view(B, T, 1, -1).transpose(1, 2)
        v = self.v(x).view(B, T, 1, -1).transpose(1, 2)
        out = linear_attn(q, k, v, self.decay).squeeze(1)
        return self.o(out)


class HCA(nn.Module):
    """
    Hybrid-Channel Attention（混合通道注意力）
    线性注意力，1 个头，衰减因子 0.999（更慢衰减 = 更长记忆）
    功能：场景级长程覆盖，压缩比 128:1
    """

    def __init__(self, dim, hd):
        super().__init__()
        self.q = nn.Linear(dim, hd, bias=False)
        self.k = nn.Linear(dim, hd, bias=False)
        self.v = nn.Linear(dim, hd, bias=False)
        self.o = nn.Linear(hd, dim, bias=False)
        self.decay = 0.999

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, 1, -1).transpose(1, 2)
        k = self.k(x).view(B, T, 1, -1).transpose(1, 2)
        v = self.v(x).view(B, T, 1, -1).transpose(1, 2)
        out = linear_attn(q, k, v, self.decay).squeeze(1)
        return self.o(out)


class SWA(nn.Module):
    """
    Sliding Window Attention（滑动窗口注意力）
    标准 softmax 因果注意力 + RoPE 旋转位置编码
    功能：保局部精度，窗口内全量注意力
    """

    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.rope = RoPE(head_dim)

    def forward(self, x):
        B, T, D = x.shape
        # 合并投影 Q/K/V（一次 GEMM）→ 拆分为 [3, B, num_heads, T, head_dim]
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B, num_heads, T, head_dim]
        # 应用旋转位置编码（对 q 和 k 使用相同的旋转角度）
        emb = self.rope(k)
        q = apply_rope(q, emb)
        k = apply_rope(k, emb)
        # 因果注意力：PyTorch 原生 SDPA，自动选 FlashAttention / memory-efficient 实现
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, -1)  # 合并多头 → [B, T, num_heads*head_dim]
        return self.proj(out)


# ── 金字塔压缩/扩展 ──

class SeqCompress(nn.Module):
    """序列下采样：平均池化，零参数量"""

    def __init__(self, kernel, stride, padding=0):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
        ).transpose(1, 2)


class SeqExpand(nn.Module):
    """序列上采样：近邻插值到目标长度"""

    def forward(self, x, target_len):
        return F.interpolate(
            x.transpose(1, 2), size=target_len, mode='nearest'
        ).transpose(1, 2)


# ── 混合注意力块（金字塔版）──

class HybridBlock(nn.Module):
    """
    混合注意力块：CSA + HCA + SWA 三分支金字塔

    结构式对齐：
      CSA — 4:1 压缩（stride=2, 重叠窗口）→ Linear Attention → 上采样
      HCA — 128:1 压缩（stride=128, 不重叠）→ Linear Attention → 上采样
      SWA — 最后 128 个 token → Softmax Attention + RoPE → 保持

    三分支输出上采样回原分辨率后，通过可学习门控加权融合。
    """

    def __init__(self, dim, num_heads, head_dim, ffn_mult=4, dropout=0.0,
                 swa_window=128, csa_ratio=2, hca_ratio=128, norm_type="rms"):
        super().__init__()
        self.swa_window = swa_window
        self.norm1 = get_norm(dim, norm_type)

        # 金字塔压缩器（平均池化，零参数量）
        self.compress_csa = SeqCompress(kernel=4, stride=csa_ratio, padding=1)  # n → n/2, 重叠
        self.compress_hca = SeqCompress(kernel=128, stride=hca_ratio)            # n → n/128, 不重叠
        self.expand = SeqExpand()

        # 三分支注意力
        self.csa = CSA(dim, head_dim)
        self.hca = HCA(dim, head_dim)
        self.swa = SWA(dim, num_heads, head_dim)

        self.gate = nn.Parameter(torch.ones(3))
        self.norm2 = get_norm(dim, norm_type)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 整个 block 在 FP32 中运行（SWA softmax + FFN GELU 在 FP16 下梯度易溢出）
        with torch.amp.autocast('cuda', enabled=False):
            h = self.norm1(x)
            B, T, D = h.shape

            # ── 金字塔压缩：三分支在不同分辨率上计算 ──
            # 短序列时跳过压缩（HCA kernel=128，T<128 时 output=0）
            h_csa = self.compress_csa(h) if T >= 4 else h       # [B, T/r_csa, D] 或 [B, T, D]
            h_hca = self.compress_hca(h) if T >= 128 else h     # [B, T/r_hca, D] 或 [B, T, D]
            h_swa = h[:, -min(self.swa_window, T):, :]          # [B, min(128,T), D]

            # ── 注意力计算 ──
            oc = self.csa(h_csa)
            oh = self.hca(h_hca)
            os = self.swa(h_swa)

            # ── 上采样到原分辨率 ──
            oc = self.expand(oc, T)
            oh = self.expand(oh, T)
            os = self.expand(os, T)

            # ── 门控融合 + 残差 ──
            g = F.softmax(self.gate, dim=0)
            x = x + self.drop(g[0] * oc + g[1] * oh + g[2] * os)

            # ── FFN + 残差 ──
            x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ── 视觉编码器 ──

class VisionEncoder(nn.Module):
    """SigLIP 视觉编码器（冻结）+ MLP 投影到 LLM dim

    采用 LLaVA 式架构：预训练 SigLIP ViT 提取图像 patch 特征，
    经两层 MLP 投影到语言模型维度后，作为前缀 token 拼接到文本嵌入前。

    为什么选 SigLIP 而非 CLIP：
      - SigLIP 使用 sigmoid loss（非 contrastive softmax），训练更高效
      - 在 ImageNet zero-shot 和检索任务上同级优于 CLIP
      - base-patch16-224 规格适中（93M 参数），不占过多显存

    冻结策略：
      - SigLIP ViT 全部参数 requires_grad=False，训练时不更新
      - 只训练 MLP 投影层 + img_pos 位置编码
      - 这样保留预训练视觉知识，只学"视觉→语言"的对齐

    SigLIP-base-patch16-224 规格：
      - 参数量：93M（冻结，不参与训练）
      - hidden_size：768
      - image_size：224×224
      - patch_size：16×16 → 14×14 = 196 个 patch token
      - 输出：[B, 196, 768]（丢弃 CLS token）→ 投影 → [B, 196, dim]
    """

    def __init__(self, model_name="google/siglip-base-patch16-224", dim=1536, freeze=True):
        super().__init__()
        from transformers import SiglipVisionModel

        self.vision = SiglipVisionModel.from_pretrained(model_name)
        vision_dim = self.vision.config.hidden_size  # 768
        image_size = self.vision.config.image_size    # 224
        patch_size = self.vision.config.patch_size     # 16
        self.num_patches = (image_size // patch_size) ** 2  # 196

        if freeze:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(vision_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.img_pos = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.freeze = freeze

    def forward(self, images):
        """
        images: [B, 3, 224, 224]（经 SiglipImageProcessor 预处理）
        → [B, 196, dim]
        """
        ctx = torch.no_grad if self.freeze else _identity_ctx
        with ctx():
            out = self.vision(pixel_values=images)
            features = out.last_hidden_state  # [B, 197, 768] (CLS + 196 patches)
            features = features[:, 1:, :]     # 丢弃 CLS token → [B, 196, 768]
        return self.proj(features) + self.img_pos  # [B, 196, dim]


class _identity_ctx:
    """空 context manager（freeze=False 时替代 torch.no_grad）"""
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ── 主模型 ──

class EdgeTransformer(nn.Module):
    """
    EdgeTransformer：端侧混合注意力语言模型

    结构：
      [VisionEncoder (可选)] → Embedding + 位置编码 → [HybridBlock × N] → Norm → Linear LM Head

    纯文本模式（vision_model=""）：
      token_ids → Embedding + pos → HybridBlock × N → LM Head
      与旧 checkpoint 完全兼容

    多模态模式（vision_model="google/siglip-base-patch16-224"）：
      images → SigLIP (frozen) → MLP → img_tokens [B, 196, dim]
      token_ids → Embedding + pos → text_tokens [B, T, dim]
      [img_tokens, text_tokens] → HybridBlock × N → LM Head
      损失只在 text 位置计算（image 位置 target=-100）

    每个 HybridBlock 内含 CSA + HCA + SWA 三分支并行注意力。
    norm_type 控制归一化层（默认 RMSNorm，可选 LayerNorm 兼容旧 checkpoint）。
    """

    def __init__(self, vocab_size=4096, dim=512, num_layers=4, num_heads=8,
                 head_dim=64, ffn_mult=4, dropout=0.0, max_seq_len=512, weight_tie=True,
                 swa_window=128, csa_ratio=2, hca_ratio=128, norm_type="rms",
                 vision_model="", vision_freeze=True):
        super().__init__()
        self.swa_window = swa_window
        self.tok = nn.Embedding(vocab_size, dim)                        # 词嵌入
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02) # 可学习位置编码
        self.layers = nn.ModuleList([
            HybridBlock(dim, num_heads, head_dim, ffn_mult, dropout,
                        swa_window, csa_ratio, hca_ratio, norm_type)
            for _ in range(num_layers)
        ])
        self.norm = get_norm(dim, norm_type)     # 最终归一化
        self.head = nn.Linear(dim, vocab_size, bias=False)  # LM 输出头
        if weight_tie:
            # 权重共享：embedding 和 LM head 共用同一矩阵，减少 ~vocab×dim 参数
            self.head.weight = self.tok.weight

        # 视觉编码器（可选）：vision_model 为空 → 纯文本模型
        self.vision = None
        self.num_image_tokens = 0
        if vision_model:
            self.vision = VisionEncoder(vision_model, dim, vision_freeze)
            self.num_image_tokens = self.vision.num_patches

        self._init_weights()
        # 注意：RMSNorm.weight 初始化为 ones（恒等映射），_init_weights 不覆盖它

    def _init_weights(self):
        """初始化权重：Linear 和 Embedding 用 N(0, 0.02)，Norm 保持 ones 初始化
        跳过冻结的预训练视觉编码器（SigLIP 权重保持不变）。"""
        skip = set()
        if self.vision is not None:
            skip = set(id(m) for m in self.vision.vision.modules())
        for m in self.modules():
            if id(m) in skip:
                continue
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, images=None):
        """
        前向传播
        x: [B, T] token ids → [B, T(+num_image_tokens), vocab_size] logits
        images: [B, 3, 224, 224] 或 None（纯文本模式）
        """
        h = self.tok(x) + self.pos[:, : x.size(1)]
        if images is not None and self.vision is not None:
            img_features = self.vision(images)  # [B, 196, dim]
            h = torch.cat([img_features, h], dim=1)  # 图像 token 前缀
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


def count_params(model):
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters())
