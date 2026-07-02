"""验证SVD分解计算是否正确"""

import torch
import numpy as np

# 创建一个已知的低秩矩阵
print("="*80)
print("测试1: 已知低秩矩阵（应该精度很高）")
print("="*80)

# 创建秩=10的矩阵 (100x100)
U_true = torch.randn(100, 10)
V_true = torch.randn(10, 100)
low_rank_matrix = U_true @ V_true  # 秩=10

# SVD分解，保留秩=10
U, S, Vh = torch.linalg.svd(low_rank_matrix, full_matrices=False)
U_k = U[:, :10]
S_k = S[:10]
Vh_k = Vh[:10, :]

# 重构
reconstructed = U_k @ torch.diag(S_k) @ Vh_k

# 计算误差
error = torch.norm(low_rank_matrix - reconstructed) / torch.norm(low_rank_matrix)
print(f"低秩矩阵(秩=10), SVD秩=10: 误差 = {error.item():.10f}")

# 测试Qwen权重
print("\n" + "="*80)
print("测试2: Qwen模型权重")
print("="*80)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

try:
    from safetensors.torch import load_file
except ImportError:
    print("需要安装safetensors")
    exit(1)

model_path = r"D:\918822019.github.io\project\ContinuePretrain\Qwen3.5-0.8B"
safetensors_path = Path(model_path) / "model.safetensors-00001-of-00001.safetensors"
state_dict = load_file(str(safetensors_path))

# 获取一个MLP层
weight = state_dict['model.language_model.layers.0.mlp.gate_proj.weight'].float()
print(f"权重形状: {weight.shape}")

# 完整SVD
U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

# 分析奇异值分布
total_energy = torch.sum(S ** 2)
print(f"总能量: {total_energy.item():.2f}")

print("\n奇异值分析:")
for k in [10, 32, 64, 128, 256, 512, 1024]:
    if k > len(S):
        break
    
    # 保留前k个奇异值
    S_k = S[:k]
    energy_k = torch.sum(S_k ** 2)
    energy_ratio = energy_k / total_energy
    
    # 重构
    U_k = U[:, :k]
    Vh_k = Vh[:k, :]
    reconstructed = U_k @ torch.diag(S_k) @ Vh_k
    
    # 计算误差
    error = torch.norm(weight - reconstructed) / torch.norm(weight)
    
    print(f"  k={k:4d}: 能量保留={energy_ratio*100:6.2f}%, 重构误差={error*100:6.2f}%")

print("\n" + "="*80)
print("结论")
print("="*80)
