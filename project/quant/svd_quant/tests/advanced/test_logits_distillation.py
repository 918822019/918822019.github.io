"""
Logits KL 蒸馏实现

核心思想：
- 不需要隐藏状态，直接用 FP16 模型和 RVQ 模型对同一批 token 计算 Logits KL 散度
- 比隐藏状态对齐更直接地约束"输出行为一致性"
- 实现成本极低：只需前向传播，无需加载完整模型的中间层 hook

优势：
1. 直接约束输出行为，而非中间表示
2. 实现简单，只需前向传播
3. 可以与 MSE Loss 结合使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import json
import time

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"


class LogitsDistillationLoss(nn.Module):
    """
    Logits 蒸馏损失
    
    使用 KL 散度约束量化模型的输出与原始模型一致
    """
    
    def __init__(self, temperature=1.0, alpha=0.5):
        """
        Args:
            temperature: 温度参数，控制 softmax 的平滑程度
            alpha: MSE 损失的权重，KL 损失的权重为 (1-alpha)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, logits_original, logits_quantized, targets=None):
        """
        计算蒸馏损失
        
        Args:
            logits_original: 原始模型的输出 logits
            logits_quantized: 量化模型的输出 logits
            targets: 目标标签（可选，用于 CE 损失）
            
        Returns:
            loss: 总损失
            mse_loss: MSE 损失
            kl_loss: KL 散度损失
        """
        # 计算 KL 散度损失
        # 使用温度缩放
        logits_original_scaled = logits_original / self.temperature
        logits_quantized_scaled = logits_quantized / self.temperature
        
        # 计算 softmax
        probs_original = F.softmax(logits_original_scaled, dim=-1)
        log_probs_quantized = F.log_softmax(logits_quantized_scaled, dim=-1)
        
        # KL 散度
        kl_loss = F.kl_div(
            log_probs_quantized,
            probs_original,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # MSE 损失
        mse_loss = F.mse_loss(logits_quantized, logits_original)
        
        # 总损失
        loss = self.alpha * mse_loss + (1 - self.alpha) * kl_loss
        
        return loss, mse_loss, kl_loss


class RouterConsistencyLoss(nn.Module):
    """
    Router 一致性损失
    
    确保 RVQ 量化后的专家，被 Router 选中的概率分布与原始模型一致
    
    增强指标：
    1. KL 散度：衡量分布差异
    2. JS 散度：更对称的分布差异度量
    3. Top-K 重合率：集合一致性
    4. Routing Weight 偏移：权重分数偏移检测
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, router_logits_original, router_logits_quantized):
        """
        计算 Router 一致性损失
        
        Args:
            router_logits_original: 原始模型的 Router logits
            router_logits_quantized: 量化模型的 Router logits
            
        Returns:
            loss: Router 一致性损失
        """
        # 计算 softmax
        probs_original = F.softmax(router_logits_original, dim=-1)
        log_probs_quantized = F.log_softmax(router_logits_quantized, dim=-1)
        
        # KL 散度
        loss = F.kl_div(
            log_probs_quantized,
            probs_original,
            reduction='batchmean'
        )
        
        return loss
    
    def compute_js_divergence(self, router_logits_original, router_logits_quantized):
        """
        计算 JS 散度（Jensen-Shannon Divergence）
        
        JS 散度是 KL 散度的对称版本，更适合衡量两个分布的差异
        
        Args:
            router_logits_original: 原始模型的 Router logits
            router_logits_quantized: 量化模型的 Router logits
            
        Returns:
            js_div: JS 散度
        """
        # 计算概率分布
        probs_original = F.softmax(router_logits_original, dim=-1)
        probs_quantized = F.softmax(router_logits_quantized, dim=-1)
        
        # 计算平均分布 M = (P + Q) / 2
        m = 0.5 * (probs_original + probs_quantized)
        
        # 计算 KL(P || M) 和 KL(Q || M)
        # 添加 epsilon 避免 log(0)
        epsilon = 1e-8
        
        # KL(P || M)
        kl_pm = torch.sum(probs_original * torch.log(probs_original + epsilon) - 
                          probs_original * torch.log(m + epsilon), dim=-1)
        
        # KL(Q || M)
        kl_qm = torch.sum(probs_quantized * torch.log(probs_quantized + epsilon) - 
                          probs_quantized * torch.log(m + epsilon), dim=-1)
        
        # JS 散度 = 0.5 * (KL(P || M) + KL(Q || M))
        js_div = 0.5 * (kl_pm + kl_qm)
        
        return js_div.mean()
    
    def compute_topk_overlap(self, router_logits_original, router_logits_quantized, k=2):
        """
        计算 Top-K 重合率
        
        Args:
            router_logits_original: 原始模型的 Router logits (batch_size, num_experts) 或 (batch_size, seq_len, num_experts)
            router_logits_quantized: 量化模型的 Router logits
            k: Top-K 值
            
        Returns:
            overlap_rate: Top-K 重合率
        """
        # 获取 Top-K 索引
        _, topk_original = torch.topk(router_logits_original, k, dim=-1)
        _, topk_quantized = torch.topk(router_logits_quantized, k, dim=-1)
        
        # 计算重合率
        # 将张量展平为 (N, k) 形状进行逐行比较
        topk_orig_flat = topk_original.reshape(-1, k)
        topk_quant_flat = topk_quantized.reshape(-1, k)
        
        overlap_count = 0
        total_count = topk_orig_flat.shape[0] * k
        
        for i in range(topk_orig_flat.shape[0]):
            set_original = set(topk_orig_flat[i].cpu().tolist())
            set_quantized = set(topk_quant_flat[i].cpu().tolist())
            overlap_count += len(set_original.intersection(set_quantized))
        
        overlap_rate = overlap_count / total_count if total_count > 0 else 0.0
        
        return overlap_rate
    
    def compute_weight_shift(self, router_logits_original, router_logits_quantized):
        """
        计算 Routing Weight 偏移
        
        检测 Top-K 集合不变但权重分数发生偏移的情况
        
        Args:
            router_logits_original: 原始模型的 Router logits
            router_logits_quantized: 量化模型的 Router logits
            
        Returns:
            weight_shift: 权重偏移统计
        """
        # 计算概率分布
        probs_original = F.softmax(router_logits_original, dim=-1)
        probs_quantized = F.softmax(router_logits_quantized, dim=-1)
        
        # 计算 L1 距离
        l1_distance = torch.abs(probs_original - probs_quantized).sum(dim=-1)
        
        # 计算相对偏移
        relative_shift = l1_distance / (probs_original.sum(dim=-1) + 1e-8)
        
        return {
            'l1_distance': l1_distance.mean().item(),
            'relative_shift': relative_shift.mean().item(),
            'max_shift': relative_shift.max().item()
        }


class DistillationTrainer:
    """
    蒸馏训练器
    
    用于训练量化模型，支持 GPU 加速和混合精度训练
    """
    
    def __init__(self, model_original, model_quantized, tokenizer, 
                 temperature=1.0, alpha=0.5, lr=1e-4, device=None):
        """
        Args:
            model_original: 原始模型
            model_quantized: 量化模型
            tokenizer: 分词器
            temperature: 温度参数
            alpha: MSE 损失的权重
            lr: 学习率
            device: 计算设备（自动检测 GPU）
        """
        # GPU 设备检测
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"\nDistillationTrainer 使用设备: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.model_original = model_original.to(self.device)
        self.model_quantized = model_quantized.to(self.device)
        self.tokenizer = tokenizer
        
        # 损失函数
        self.logits_loss = LogitsDistillationLoss(temperature, alpha).to(self.device)
        self.router_loss = RouterConsistencyLoss().to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model_quantized.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        # 混合精度训练（仅 GPU）
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            torch.backends.cudnn.benchmark = True
            print("启用混合精度训练 (FP16/FP32)")
    
    def train_step(self, input_ids, attention_mask=None):
        """
        训练一步（支持 GPU 混合精度）
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            
        Returns:
            loss: 总损失
            mse_loss: MSE 损失
            kl_loss: KL 散度损失
            router_metrics: Router 一致性指标（如果有）
        """
        # 移至设备
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 前向传播（原始模型）
        with torch.no_grad():
            outputs_original = self.model_original(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits_original = outputs_original.logits
            # 获取 Router logits（如果有）
            router_logits_original = getattr(outputs_original, 'router_logits', None)
        
        # 前向传播（量化模型）- GPU 混合精度
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                outputs_quantized = self.model_quantized(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                logits_quantized = outputs_quantized.logits
                router_logits_quantized = getattr(outputs_quantized, 'router_logits', None)
                
                # 计算损失
                loss, mse_loss, kl_loss = self.logits_loss(
                    logits_original,
                    logits_quantized
                )
                
                # 如果有 Router logits，添加 Router 一致性损失
                router_metrics = None
                if router_logits_original is not None and router_logits_quantized is not None:
                    router_loss = self.router_loss(
                        router_logits_original,
                        router_logits_quantized
                    )
                    loss = loss + 0.1 * router_loss
                    
                    # 计算增强指标
                    js_div = self.router_loss.compute_js_divergence(
                        router_logits_original,
                        router_logits_quantized
                    )
                    topk_overlap = self.router_loss.compute_topk_overlap(
                        router_logits_original,
                        router_logits_quantized
                    )
                    weight_shift = self.router_loss.compute_weight_shift(
                        router_logits_original,
                        router_logits_quantized
                    )
                    
                    router_metrics = {
                        'router_loss': router_loss.item(),
                        'js_divergence': js_div.item(),
                        'topk_overlap': topk_overlap,
                        'weight_shift': weight_shift
                    }
            
            # 混合精度反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # CPU 模式
            outputs_quantized = self.model_quantized(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits_quantized = outputs_quantized.logits
            router_logits_quantized = getattr(outputs_quantized, 'router_logits', None)
            
            # 计算损失
            loss, mse_loss, kl_loss = self.logits_loss(
                logits_original,
                logits_quantized
            )
            
            # 如果有 Router logits，添加 Router 一致性损失
            router_metrics = None
            if router_logits_original is not None and router_logits_quantized is not None:
                router_loss = self.router_loss(
                    router_logits_original,
                    router_logits_quantized
                )
                loss = loss + 0.1 * router_loss
                
                # 计算增强指标
                js_div = self.router_loss.compute_js_divergence(
                    router_logits_original,
                    router_logits_quantized
                )
                topk_overlap = self.router_loss.compute_topk_overlap(
                    router_logits_original,
                    router_logits_quantized
                )
                weight_shift = self.router_loss.compute_weight_shift(
                    router_logits_original,
                    router_logits_quantized
                )
                
                router_metrics = {
                    'router_loss': router_loss.item(),
                    'js_divergence': js_div.item(),
                    'topk_overlap': topk_overlap,
                    'weight_shift': weight_shift
                }
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item(), mse_loss.item(), kl_loss.item(), router_metrics
    
    def evaluate(self, input_ids, attention_mask=None):
        """
        评估（支持 GPU）
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            
        Returns:
            loss: 总损失
            mse_loss: MSE 损失
            kl_loss: KL 散度损失
        """
        # 移至设备
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            # 前向传播（原始模型）
            outputs_original = self.model_original(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits_original = outputs_original.logits
            
            # 前向传播（量化模型）
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs_quantized = self.model_quantized(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits_quantized = outputs_quantized.logits
                    
                    # 计算损失
                    loss, mse_loss, kl_loss = self.logits_loss(
                        logits_original,
                        logits_quantized
                    )
            else:
                outputs_quantized = self.model_quantized(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits_quantized = outputs_quantized.logits
                
                # 计算损失
                loss, mse_loss, kl_loss = self.logits_loss(
                    logits_original,
                    logits_quantized
                )
        
        return loss.item(), mse_loss.item(), kl_loss.item()


def test_distillation_concept():
    """
    测试蒸馏概念（不加载完整模型，支持 GPU）
    """
    print("=" * 70)
    print("Logits KL 蒸馏概念测试")
    print("=" * 70)
    
    # GPU 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n计算设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA 版本: {torch.version.cuda}")
    
    # 创建模拟数据
    batch_size = 4
    seq_len = 128
    vocab_size = 32000
    
    # 模拟原始模型输出
    logits_original = torch.randn(batch_size, seq_len, vocab_size, device=device)
    
    # 模拟量化模型输出（添加噪声）
    logits_quantized = logits_original + torch.randn_like(logits_original) * 0.1
    
    # 创建损失函数
    loss_fn = LogitsDistillationLoss(temperature=1.0, alpha=0.5).to(device)
    
    # 计算损失
    loss, mse_loss, kl_loss = loss_fn(logits_original, logits_quantized)
    
    print(f"\n损失计算结果:")
    print(f"  总损失: {loss.item():.4f}")
    print(f"  MSE 损失: {mse_loss.item():.4f}")
    print(f"  KL 散度损失: {kl_loss.item():.4f}")
    
    # 测试不同温度
    print("\n不同温度下的 KL 散度:")
    temperatures = [0.5, 1.0, 2.0, 4.0]
    for temp in temperatures:
        loss_fn_temp = LogitsDistillationLoss(temperature=temp, alpha=0.5).to(device)
        _, _, kl = loss_fn_temp(logits_original, logits_quantized)
        print(f"  温度={temp}: KL={kl.item():.4f}")
    
    # 测试 Router 一致性损失
    print("\nRouter 一致性损失测试:")
    num_experts = 8
    router_logits_original = torch.randn(batch_size, num_experts, device=device)
    router_logits_quantized = router_logits_original + torch.randn_like(router_logits_original) * 0.2
    
    router_loss_fn = RouterConsistencyLoss().to(device)
    router_loss = router_loss_fn(router_logits_original, router_logits_quantized)
    print(f"  Router 一致性损失: {router_loss.item():.4f}")
    
    # 测试 JS 散度
    js_div = router_loss_fn.compute_js_divergence(router_logits_original, router_logits_quantized)
    print(f"  JS 散度: {js_div.item():.4f}")
    
    # 测试 Top-K 重合率
    topk_overlap = router_loss_fn.compute_topk_overlap(router_logits_original, router_logits_quantized, k=2)
    print(f"  Top-2 重合率: {topk_overlap:.4f}")
    
    # 测试权重偏移
    weight_shift = router_loss_fn.compute_weight_shift(router_logits_original, router_logits_quantized)
    print(f"  权重偏移:")
    print(f"    L1 距离: {weight_shift['l1_distance']:.4f}")
    print(f"    相对偏移: {weight_shift['relative_shift']:.4f}")
    print(f"    最大偏移: {weight_shift['max_shift']:.4f}")
    
    # 分析蒸馏效果
    print("\n蒸馏效果分析:")
    print("  1. MSE 损失约束数值一致性")
    print("  2. KL 散度约束分布一致性")
    print("  3. Router 一致性损失约束专家选择行为")
    print("  4. JS 散度提供更对称的分布差异度量")
    print("  5. Top-K 重合率检测集合一致性")
    print("  6. 权重偏移检测分数偏移问题")
    print("  7. 七者结合确保量化模型行为与原始模型一致")
    
    # GPU 显存监控
    if device.type == "cuda":
        print(f"\nGPU 显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"GPU 显存预留: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    return {
        "loss": loss.item(),
        "mse_loss": mse_loss.item(),
        "kl_loss": kl_loss.item(),
        "router_loss": router_loss.item(),
        "js_divergence": js_div.item(),
        "topk_overlap": topk_overlap,
        "weight_shift": weight_shift
    }


def create_distillation_training_script():
    """
    创建蒸馏训练脚本模板
    """
    script_content = '''
# 蒸馏训练脚本模板
# 注意：需要安装 transformers 和 torch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def train_with_distillation():
    """
    使用蒸馏训练量化模型
    """
    # 1. 加载原始模型
    model_original = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 2. 加载量化模型（需要修改为你的量化模型）
    # model_quantized = YourQuantizedModel(...)
    
    # 3. 创建蒸馏训练器
    trainer = DistillationTrainer(
        model_original=model_original,
        model_quantized=model_quantized,
        tokenizer=tokenizer,
        temperature=1.0,
        alpha=0.5,
        lr=1e-4
    )
    
    # 4. 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss, mse_loss, kl_loss = trainer.train_step(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            print(f"Epoch {epoch}, Loss: {loss:.4f}, MSE: {mse_loss:.4f}, KL: {kl_loss:.4f}")

if __name__ == "__main__":
    train_with_distillation()
'''
    
    # 保存脚本
    with open("distillation_training_template.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("\n蒸馏训练脚本模板已保存到: distillation_training_template.py")
    print("请根据你的量化模型进行修改。")


if __name__ == "__main__":
    # 运行概念测试
    results = test_distillation_concept()
    
    # 创建训练脚本模板
    create_distillation_training_script()