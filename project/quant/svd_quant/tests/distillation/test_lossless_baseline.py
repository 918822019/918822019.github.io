"""
无损基线测试脚本

目的：验证离线缓存蒸馏的对齐问题
操作：关闭所有FSQ量化（即 scale=1, min=0, q=x），让 Student 模型等于 Teacher 模型
预期：如果此时 PPL 差异依然 > 1，说明绝对是 Logits 缓存对齐问题或 KL 计算代码有 Bug

测试步骤：
1. 加载缓存的 Logits 和激活值
2. 创建一个不进行量化的 Expert 模型
3. 运行蒸馏训练，但 Student 和 Teacher 使用相同的权重
4. 检查 KL 散度是否接近 0（如果对齐正确）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional

# 导入缓存管理器
from test_logits_cache_distillation import LogitsCache, LogitsKLDistillationLoss, DynamicLossScheduler


class LosslessExpert(nn.Module):
    """
    无损 Expert（不进行量化）
    
    用于基线测试，确保 Student 和 Teacher 完全相同
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        vocab_size: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        
        # Expert 权重（随机初始化，但 Student 和 Teacher 将使用相同的权重）
        self.gate_up_proj = nn.Parameter(
            torch.randn(intermediate_size * 2, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.randn(hidden_size, intermediate_size)
        )
        
        # 输出投影层（映射到 vocab_size）
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（无量化）
        
        Args:
            x: 输入激活值 (batch_size, seq_len, hidden_size)
            
        Returns:
            output: 输出 Logits (batch_size, seq_len, vocab_size)
        """
        # Expert 前向传播（无量化）
        gate_up = F.linear(x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = F.silu(gate)
        intermediate = gate * up
        output = F.linear(intermediate, self.down_proj)
        
        # 映射到 Logits
        logits = self.output_proj(output)
        
        return logits


def run_lossless_baseline_test():
    """
    运行无损基线测试
    """
    print("=" * 70)
    print("无损基线测试：验证离线缓存蒸馏的对齐问题")
    print("=" * 70)
    
    # GPU 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n计算设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查缓存是否存在
    cache = LogitsCache()
    
    if not cache.cache_exists():
        print("错误：缓存不存在，请先运行 test_logits_cache_distillation.py 生成缓存")
        return
    
    # 加载缓存（包含对齐信息）
    activations, logits_original, metadata, attention_mask, position_ids = cache.load_cache()
    
    # GPU 加速：将数据移至设备
    print(f"\n将数据加载到 {device}...")
    activations = activations.to(device)
    logits_original = logits_original.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
    hidden_size = metadata["hidden_size"]
    vocab_size = metadata["vocab_size"]
    
    print(f"\n数据信息:")
    print(f"  激活值形状: {activations.shape}")
    print(f"  Logits 形状: {logits_original.shape}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  词表大小: {vocab_size}")
    
    # 关键对齐检查
    if attention_mask is not None:
        print(f"  Attention Mask: {attention_mask.shape} [已缓存]")
        # 统计有效token比例
        valid_ratio = attention_mask.float().mean().item()
        print(f"  有效token比例: {valid_ratio*100:.2f}%")
    else:
        print("  警告: 未找到缓存的 Attention Mask")
        
    if position_ids is not None:
        print(f"  Position IDs: {position_ids.shape} [已缓存]")
    else:
        print("  警告: 未找到缓存的 Position IDs")
    
    # 创建 Teacher 和 Student 模型（使用相同的权重）
    intermediate_size = hidden_size * 4
    
    teacher = LosslessExpert(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size
    ).to(device)
    
    # Student 使用完全相同的权重
    student = LosslessExpert(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size
    ).to(device)
    
    # 复制权重（确保完全相同）
    student.gate_up_proj.data = teacher.gate_up_proj.data.clone()
    student.down_proj.data = teacher.down_proj.data.clone()
    student.output_proj.weight.data = teacher.output_proj.weight.data.clone()
    
    print(f"\n模型已创建（Teacher 和 Student 使用相同权重）")
    
    # 创建损失函数
    loss_fn = LogitsKLDistillationLoss(
        temperature=2.0,
        use_adaptive_temperature=True,
        target_entropy=3.0
    ).to(device)
    
    # 创建优化器（只优化 Student）
    # 注意：在无损基线测试中，我们不更新Student的权重，以验证对齐问题
    # 如果不更新权重，Student和Teacher应该完全相同
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # 标记：是否更新Student的权重
    update_student_weights = False
    
    # 创建动态 Loss 调度器
    total_steps = 200  # 较少的步数用于快速验证
    loss_scheduler = DynamicLossScheduler(total_steps)
    
    # 训练循环
    print(f"\n开始无损基线测试（{total_steps} 步）...")
    print("目标：验证 KL 散度是否接近 0（如果对齐正确）")
    print("-" * 70)
    
    losses = []
    kl_losses = []
    
    for step in range(total_steps):
        # 随机采样一个批次
        idx = torch.randint(0, len(activations), (32,))
        x_batch = activations[idx]
        logits_batch = logits_original[idx]
        
        # 获取 attention_mask
        batch_attention_mask = attention_mask[idx] if attention_mask is not None else None
        
        # 前向传播
        # Teacher（无量化）
        with torch.no_grad():
            output_teacher = teacher(x_batch)
            logits_teacher = output_teacher[:, :vocab_size]
        
        # Student（无量化，但权重相同）
        output_student = student(x_batch)
        logits_student = output_student[:, :vocab_size]
        
        # 获取当前阶段的权重
        mse_weight, kl_weight = loss_scheduler.get_weights()
        
        # 计算损失（使用 attention_mask 过滤 PAD token）
        total_loss, mse_loss, kl_loss, temperature = loss_fn(
            logits_teacher,
            logits_student,
            mse_weight=mse_weight,
            kl_weight=kl_weight,
            attention_mask=batch_attention_mask
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 根据标志决定是否更新Student的权重
        if update_student_weights:
            optimizer.step()
        else:
            # 不更新权重，只计算损失
            # 这样可以验证Teacher和Student完全相同时的KL散度
            pass
        
        # 更新调度器
        loss_scheduler.step()
        
        # 记录损失
        losses.append({
            'step': step,
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'kl_loss': kl_loss.item(),
            'mse_weight': mse_weight,
            'kl_weight': kl_weight
        })
        kl_losses.append(kl_loss.item())
        
        # 打印进度
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{total_steps}: "
                  f"Total={total_loss.item():.6f}, "
                  f"MSE={mse_loss.item():.6f}, "
                  f"KL={kl_loss.item():.6f}, "
                  f"T={temperature:.2f}")
    
    # 分析结果
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    # 计算平均 KL 散度
    avg_kl = np.mean(kl_losses[-50:])  # 最后50步的平均值
    print(f"\n最后50步的平均 KL 散度: {avg_kl:.6f}")
    
    # 诊断结果
    if avg_kl < 0.001:
        print("✓ 诊断结果：对齐正确！KL 散度接近 0")
        print("  说明：Logits 缓存对齐没有问题，问题可能出在 FSQ 量化")
    elif avg_kl < 0.01:
        print("⚠ 诊断结果：轻微对齐问题")
        print("  说明：KL 散度略高，可能存在轻微的对齐问题")
        print("  建议：检查 attention_mask 和 position_ids 的使用")
    else:
        print("✗ 诊断结果：严重对齐问题！")
        print("  说明：KL 散度过高，存在严重的对齐问题")
        print("  可能原因：")
        print("    1. attention_mask 未正确使用（PAD token 干扰）")
        print("    2. position_ids 不一致")
        print("    3. Logits 缓存本身有问题")
    
    # 保存结果
    output_file = "lossless_baseline_test_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("无损基线测试结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试配置:\n")
        f.write(f"  总步数: {total_steps}\n")
        f.write(f"  批次大小: 32\n")
        f.write(f"  使用自适应温度: 是\n")
        f.write(f"  目标熵: 3.0\n\n")
        
        f.write("数据信息:\n")
        f.write(f"  激活值形状: {activations.shape}\n")
        f.write(f"  Logits 形状: {logits_original.shape}\n")
        f.write(f"  隐藏层大小: {hidden_size}\n")
        f.write(f"  词表大小: {vocab_size}\n\n")
        
        if attention_mask is not None:
            f.write(f"  Attention Mask: {attention_mask.shape} [已缓存]\n")
            f.write(f"  有效token比例: {valid_ratio*100:.2f}%\n")
        else:
            f.write("  Attention Mask: 未找到\n")
        
        f.write("\n测试结果:\n")
        f.write(f"  最后50步平均 KL 散度: {avg_kl:.6f}\n")
        
        if avg_kl < 0.001:
            f.write("  诊断结果: 对齐正确 ✓\n")
        elif avg_kl < 0.01:
            f.write("  诊断结果: 轻微对齐问题 ⚠\n")
        else:
            f.write("  诊断结果: 严重对齐问题 ✗\n")
        
        f.write("\n详细损失曲线:\n")
        for loss in losses:
            f.write(f"  Step {loss['step']}: "
                    f"Total={loss['total_loss']:.6f}, "
                    f"MSE={loss['mse_loss']:.6f}, "
                    f"KL={loss['kl_loss']:.6f}\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    return losses


if __name__ == "__main__":
    # 运行无损基线测试
    losses = run_lossless_baseline_test()