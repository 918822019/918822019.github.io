"""
FSQ (Finite Scalar Quantization) 实现 - 修正版本

核心思想：
- 将连续值映射到有限的整数网格
- 无需维护显式码本向量
- 通过归一化和取整实现量化
- 避免码本坍塌和能量泄漏

修正：
1. 先对权重进行归一化到 [0, 1] 范围
2. 然后映射到 [0, levels-1] 并取整
3. 最后反归一化回原始范围
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import json
import time
from sklearn.cluster import KMeans

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"


class FSQQuantizerV2(nn.Module):
    """
    Finite Scalar Quantization (FSQ) - 修正版本
    
    将连续值映射到有限的整数网格
    """
    
    def __init__(self, levels=8):
        """
        Args:
            levels: 量化级别数（例如 8 表示 8 个级别）
        """
        super().__init__()
        self.levels = levels
        
        # 计算总的码本大小
        self.codebook_size = levels
        
        # 存储最小值和最大值（用于归一化）
        self.min_val = None
        self.max_val = None
    
    def fit(self, x):
        """
        计算归一化参数
        
        Args:
            x: 输入张量
        """
        self.min_val = x.min()
        self.max_val = x.max()
        
        # 避免除零
        if self.max_val - self.min_val < 1e-8:
            self.max_val = self.min_val + 1e-8
    
    def quantize(self, x):
        """
        量化操作
        
        Args:
            x: 输入张量
            
        Returns:
            x_quantized: 量化后的张量
            indices: 量化索引
        """
        if self.min_val is None or self.max_val is None:
            self.fit(x)
        
        # 归一化到 [0, 1]
        x_normalized = (x - self.min_val) / (self.max_val - self.min_val)
        
        # 映射到 [0, levels-1] 并取整
        levels = float(self.levels)
        x_scaled = x_normalized * (levels - 1)
        
        # 取整（使用 straight-through estimator）
        x_rounded = torch.round(x_scaled)
        
        # 计算索引（用于存储）
        indices = x_rounded.long()
        
        # 反归一化：映射回原始范围
        x_normalized_back = x_rounded / (levels - 1)
        x_quantized = x_normalized_back * (self.max_val - self.min_val) + self.min_val
        
        return x_quantized, indices
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            x_quantized: 量化后的张量
            indices: 量化索引
        """
        # 确保输入是浮点型
        x = x.float()
        
        # 量化
        x_quantized, indices = self.quantize(x)
        
        # 使用 straight-through estimator 传递梯度
        x_quantized = x + (x_quantized - x).detach()
        
        return x_quantized, indices
    
    def get_codebook(self):
        """
        获取码本（用于分析）
        
        Returns:
            codebook: 码本张量
        """
        # 生成所有可能的索引
        indices = torch.arange(self.levels)
        
        # 反归一化得到码本
        levels = float(self.levels)
        codebook_normalized = indices.float() / (levels - 1)
        codebook = codebook_normalized * (self.max_val - self.min_val) + self.min_val
        
        return codebook


class MultiLevelFSQV2(nn.Module):
    """
    多级 FSQ（类似 RVQ 的层级结构）- 修正版本
    
    每一级使用独立的 FSQ 量化残差
    """
    
    def __init__(self, num_levels=4, levels=8):
        """
        Args:
            num_levels: 级数
            levels: 每级的量化级别数
        """
        super().__init__()
        self.num_levels = num_levels
        
        # 创建多级 FSQ
        self.fsqs = nn.ModuleList()
        for _ in range(num_levels):
            self.fsqs.append(FSQQuantizerV2(levels=levels))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            reconstructed: 重建后的张量
            all_indices: 各级的量化索引
            all_quantized: 各级的量化结果
        """
        residual = x.clone()
        all_indices = []
        all_quantized = []
        
        for fsq in self.fsqs:
            # 拟合残差
            fsq.fit(residual)
            
            # 量化
            quantized, indices = fsq(residual)
            all_indices.append(indices)
            all_quantized.append(quantized)
            residual = residual - quantized
        
        reconstructed = sum(all_quantized)
        
        return reconstructed, all_indices, all_quantized


def load_expert_weight(file_path, expert_name, expert_idx=0):
    """
    加载指定Expert的权重
    """
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        for key in keys:
            if expert_name in key:
                weight = f.get_tensor(key)
                print(f"加载权重: {key}, 形状: {weight.shape}")
                
                if weight.ndim == 3:
                    expert_weight = weight[expert_idx].float()
                    print(f"选择专家 {expert_idx}, 形状: {expert_weight.shape}")
                    return expert_weight, key
                else:
                    return weight.float(), key
    
    return None, None


def test_fsq_v2():
    """
    测试 FSQ V2 量化
    """
    print("=" * 70)
    print("FSQ V2 (Finite Scalar Quantization) 测试")
    print("=" * 70)
    
    # 加载模型索引
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    
    # 选择一个 MoE 专家层
    moe_keys = [k for k in weight_map.keys() if "experts" in k]
    if not moe_keys:
        moe_keys = [k for k in weight_map.keys() if "gate" in k or "up" in k or "down" in k]
    
    selected_key = moe_keys[0]
    file_name = weight_map[selected_key]
    file_path = f"{model_path}/{file_name}"
    
    # 加载权重
    expert_weight, weight_key = load_expert_weight(file_path, selected_key)
    
    # 测试不同配置
    configs = [
        {"levels": 8, "name": "FSQ-8"},
        {"levels": 16, "name": "FSQ-16"},
        {"levels": 32, "name": "FSQ-32"},
        {"levels": 64, "name": "FSQ-64"},
        {"levels": 128, "name": "FSQ-128"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"配置: {config['name']}")
        print(f"{'='*50}")
        
        # 创建 FSQ 量化器
        fsq = FSQQuantizerV2(levels=config["levels"])
        
        # 拟合数据
        fsq.fit(expert_weight)
        
        # 量化
        with torch.no_grad():
            quantized, indices = fsq(expert_weight)
        
        # 计算误差
        error = torch.norm(expert_weight - quantized) / torch.norm(expert_weight)
        print(f"量化误差: {error.item()*100:.2f}%")
        
        # 计算压缩比
        # 每个元素需要 log2(levels) 位
        bits_per_element = np.log2(fsq.codebook_size)
        original_bits = expert_weight.numel() * 32  # float32
        compressed_bits = expert_weight.numel() * bits_per_element
        compression_ratio = original_bits / compressed_bits
        print(f"压缩比: {compression_ratio:.2f}x")
        print(f"码本大小: {fsq.codebook_size}")
        print(f"每元素位数: {bits_per_element:.2f}")
        
        # 分析量化索引的分布
        print(f"索引形状: {indices.shape}")
        print(f"索引范围: [{indices.min().item()}, {indices.max().item()}]")
        
        # 分析码本
        codebook = fsq.get_codebook()
        print(f"码本范围: [{codebook.min().item():.4f}, {codebook.max().item():.4f}]")
        
        results[config["name"]] = {
            "error": error.item(),
            "compression_ratio": compression_ratio,
            "codebook_size": fsq.codebook_size,
            "bits_per_element": bits_per_element
        }
    
    # 测试多级 FSQ
    print(f"\n{'='*50}")
    print(f"多级 FSQ 测试")
    print(f"{'='*50}")
    
    multi_fsq = MultiLevelFSQV2(num_levels=4, levels=8)
    
    with torch.no_grad():
        reconstructed, all_indices, all_quantized = multi_fsq(expert_weight)
    
    # 计算各级误差
    print("\n各级重建误差:")
    for i, quantized in enumerate(all_quantized):
        partial_recon = sum(all_quantized[:i+1])
        error = torch.norm(expert_weight - partial_recon) / torch.norm(expert_weight)
        print(f"  L{i+1}: {error.item()*100:.2f}%")
    
    # 计算能量分布
    print("\n各级能量分布:")
    total_energy = sum(torch.norm(q).item()**2 for q in all_quantized)
    for i, quantized in enumerate(all_quantized):
        energy = torch.norm(quantized).item()**2
        ratio = energy / total_energy
        print(f"  L{i+1}: {ratio*100:.2f}%")
    
    # 与标准 RVQ 对比
    print(f"\n{'='*50}")
    print(f"FSQ vs 标准 RVQ 对比")
    print(f"{'='*50}")
    
    # 标准 RVQ
    vectors = expert_weight.numpy()
    residual = vectors.copy()
    rvq_quantized = []
    
    for level in range(4):
        actual_codebook_size = min(256, len(residual))
        kmeans = KMeans(n_clusters=actual_codebook_size, max_iter=50, random_state=42, n_init=10)
        kmeans.fit(residual)
        
        labels = kmeans.predict(residual)
        quantized = kmeans.cluster_centers_[labels]
        rvq_quantized.append(quantized)
        
        residual = residual - quantized
    
    # 计算 RVQ 能量分布
    print("\n标准 RVQ 能量分布:")
    rvq_total_energy = sum(np.linalg.norm(q)**2 for q in rvq_quantized)
    for i, quantized in enumerate(rvq_quantized):
        energy = np.linalg.norm(quantized)**2
        ratio = energy / rvq_total_energy
        print(f"  L{i+1}: {ratio*100:.2f}%")
    
    # 保存结果
    output_file = "fsq_v2_test_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("FSQ V2 (Finite Scalar Quantization) 测试结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试权重: " + selected_key + "\n")
        f.write("权重形状: " + str(expert_weight.shape) + "\n\n")
        
        f.write("不同配置结果:\n")
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  量化误差: {result['error']*100:.2f}%\n")
            f.write(f"  压缩比: {result['compression_ratio']:.2f}x\n")
            f.write(f"  码本大小: {result['codebook_size']}\n")
            f.write(f"  每元素位数: {result['bits_per_element']:.2f}\n")
        
        f.write("\n多级 FSQ 结果:\n")
        f.write("各级重建误差:\n")
        for i, quantized in enumerate(all_quantized):
            partial_recon = sum(all_quantized[:i+1])
            error = torch.norm(expert_weight - partial_recon) / torch.norm(expert_weight)
            f.write(f"  L{i+1}: {error.item()*100:.2f}%\n")
        
        f.write("\n各级能量分布:\n")
        for i, quantized in enumerate(all_quantized):
            energy = torch.norm(quantized).item()**2
            ratio = energy / total_energy
            f.write(f"  L{i+1}: {ratio*100:.2f}%\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    # 运行测试
    results = test_fsq_v2()