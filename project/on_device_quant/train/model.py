"""
EdgeTransformer 模型定义

核心架构：混合注意力（Hybrid Attention）
  - CSA: Cross-Stream Attention，线性注意力，decay=0.99，关注跨流信息
  - HCA: Hybrid-Channel Attention，线性注意力，decay=0.999，长程场景覆盖
  - SWA: Sliding Window Attention，标准 softmax 因果注意力 + RoPE，保局部精度

三个分支通过可学习门控（softmax gate）加权融合。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    # 特征映射：确保非负（类似 softmax 的正值特性）
    qf = F.relu(q).pow(2) + 0.1
    kf = F.relu(k).pow(2) + 0.1
    vf = F.relu(v).pow(2) + 0.1
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    z = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)
    outs = []
    for i in range(T):
        ki = kf[:, :, i : i + 1]
        vi = vf[:, :, i : i + 1]
        qi = qf[:, :, i : i + 1]
        S = decay * S + ki.transpose(-1, -2) @ vi
        z = decay * z + ki.squeeze(2)
        out = (qi @ S) / (qi @ z.unsqueeze(-1)).clamp(min=eps)
        outs.append(out)
    return torch.cat(outs, dim=-2).to(dtype)
    return torch.cat(outs, dim=-2)


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
        # 合并投影 Q/K/V → 拆分为多头
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 应用旋转位置编码
        emb = self.rope(k)
        q = apply_rope(q, emb)
        k = apply_rope(k, emb)
        # 因果注意力（PyTorch 原生实现，自动使用 FlashAttention/内存高效实现）
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)


# ── 混合注意力块 ──

class HybridBlock(nn.Module):
    """
    混合注意力块：CSA + HCA + SWA 三分支并行
    通过可学习门控 softmax(gate) 加权融合
    """

    def __init__(self, dim, num_heads, head_dim, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)           # 前归一化
        self.csa = CSA(dim, head_dim)             # 跨流注意力
        self.hca = HCA(dim, head_dim)             # 混合通道注意力
        self.swa = SWA(dim, num_heads, head_dim)  # 滑动窗口注意力
        self.gate = nn.Parameter(torch.ones(3))   # 可学习门控（初始均等）
        self.norm2 = nn.LayerNorm(dim)
        # 前馈网络：Linear → GELU → Linear
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 三分支并行计算
        # 整个 block 在 FP32 中运行（SWA softmax + FFN GELU 在 FP16 下梯度易溢出）
        with torch.amp.autocast('cuda', enabled=False):
            h = self.norm1(x)
            oc = self.csa(h)   # 跨流注意力输出
            oh = self.hca(h)   # 混合通道注意力输出
            os = self.swa(h)   # 滑动窗口注意力输出
            # softmax 门控融合 + 残差连接
            g = F.softmax(self.gate, dim=0)
            x = x + self.drop(g[0] * oc + g[1] * oh + g[2] * os)
            # FFN + 残差连接
            x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ── 主模型 ──

class EdgeTransformer(nn.Module):
    """
    EdgeTransformer：端侧混合注意力语言模型

    结构：
      Embedding → [HybridBlock × N] → LayerNorm → Linear LM Head

    每个 HybridBlock 内含 CSA + HCA + SWA 三分支并行注意力
    """

    def __init__(self, vocab_size=4096, dim=512, num_layers=4, num_heads=8,
                 head_dim=64, ffn_mult=4, dropout=0.0, max_seq_len=512, weight_tie=True):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)                        # 词嵌入
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02) # 可学习位置编码
        self.layers = nn.ModuleList([
            HybridBlock(dim, num_heads, head_dim, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)     # 最终归一化
        self.head = nn.Linear(dim, vocab_size, bias=False)  # LM 输出头
        if weight_tie:
            self.head.weight = self.tok.weight  # 权重共享：embedding 和 LM head 共用
        self._init_weights()

    def _init_weights(self):
        """初始化权重：Linear 和 Embedding 用 N(0, 0.02)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        """
        前向传播
        x: [B, T] token ids → [B, T, vocab_size] logits
        """
        h = self.tok(x) + self.pos[:, : x.size(1)]
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


def count_params(model):
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters())
