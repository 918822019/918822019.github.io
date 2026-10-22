"""
混合粒度 RVQ 实现

核心思想：
- 对权重矩阵做分块 SVD，找出哪些块的奇异值衰减快（低秩），哪些块衰减慢（满秩）
- 低秩块：用 2 级 RVQ + 大码本（捕获主成分）
- 满秩块：用 4 级 RVQ + 小码本（或直接用 INT4/HQQ）
- 让 RVQ 的深度适应权重的局部秩，而非全局预设

优势：
1. 根据权重的局部特性自适应调整量化策略
2. 低秩区域使用更少的级别，满秩区域使用更多的级别
3. 整体压缩效率更高
"""

import torch
import torch.nn as nn
import numpy as np
from safetensors import safe_open
import json
import time
from sklearn.cluster import KMeans

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"


def analyze_block_rank(weight_matrix, block_size=64, rank_threshold=0.5):
    """
    分析权重矩阵的分块秩
    
    Args:
        weight_matrix: 权重矩阵
        block_size: 分块大小
        rank_threshold: 秩阈值（相对于块大小的比例）
        
    Returns:
        block_ranks: 每个块的秩
        low_rank_blocks: 低秩块的索引
        high_rank_blocks: 满秩块的索引
    """
    rows, cols = weight_matrix.shape
    
    # 计算块的数量
    n_blocks_row = (rows + block_size - 1) // block_size
    n_blocks_col = (cols + block_size - 1) // block_size
    
    block_ranks = []
    block_indices = []
    
    for i in range(n_blocks_row):
        for j in range(n_blocks_col):
            # 提取块
            start_row = i * block_size
            end_row = min((i + 1) * block_size, rows)
            start_col = j * block_size
            end_col = min((j + 1) * block_size, cols)
            
            block = weight_matrix[start_row:end_row, start_col:end_col]
            
            # 计算秩（使用 SVD）
            try:
                U, S, Vh = np.linalg.svd(block, full_matrices=False)
                
                # 计算能量保留 95% 所需的秩
                total_energy = np.sum(S**2)
                cumulative_energy = np.cumsum(S**2)
                rank_95 = np.searchsorted(cumulative_energy, 0.95 * total_energy) + 1
                
                # 相对秩
                relative_rank = rank_95 / min(block.shape)
                
                block_ranks.append(relative_rank)
                block_indices.append((i, j, start_row, end_row, start_col, end_col))
            except:
                # 如果 SVD 失败，假设是满秩
                block_ranks.append(1.0)
                block_indices.append((i, j, start_row, end_row, start_col, end_col))
    
    block_ranks = np.array(block_ranks)
    
    # 分类低秩和满秩块
    low_rank_mask = block_ranks < rank_threshold
    high_rank_mask = ~low_rank_mask
    
    low_rank_blocks = [block_indices[i] for i in range(len(block_indices)) if low_rank_mask[i]]
    high_rank_blocks = [block_indices[i] for i in range(len(block_indices)) if high_rank_mask[i]]
    
    return block_ranks, low_rank_blocks, high_rank_blocks


class HybridRVQ:
    """
    混合粒度 RVQ
    
    根据块的秩自适应调整 RVQ 的深度和码本大小
    """
    
    def __init__(self, block_size=64, rank_threshold=0.5):
        """
        Args:
            block_size: 分块大小
            rank_threshold: 秩阈值
        """
        self.block_size = block_size
        self.rank_threshold = rank_threshold
    
    def quantize_block(self, block, is_low_rank):
        """
        量化单个块
        
        Args:
            block: 权重块
            is_low_rank: 是否为低秩块
            
        Returns:
            quantized: 量化后的块
            num_levels: 使用的级别数
            codebook_size: 码本大小
        """
        if is_low_rank:
            # 低秩块：2 级 RVQ + 大码本
            num_levels = 2
            codebook_size = 512
        else:
            # 满秩块：4 级 RVQ + 小码本
            num_levels = 4
            codebook_size = 128
        
        # 转换为向量集合（每行作为一个向量）
        vectors = block.copy()
        residual = vectors.copy()
        all_quantized = []
        
        for level in range(num_levels):
            # 确保码本大小不超过向量数量
            actual_codebook_size = min(codebook_size, len(residual))
            
            # K-means 聚类
            kmeans = KMeans(
                n_clusters=actual_codebook_size,
                max_iter=50,
                random_state=42,
                n_init=10
            )
            kmeans.fit(residual)
            
            labels = kmeans.predict(residual)
            quantized = kmeans.cluster_centers_[labels]
            all_quantized.append(quantized)
            
            residual = residual - quantized
        
        # 重建
        reconstructed = sum(all_quantized)
        
        return reconstructed, num_levels, codebook_size
    
    def quantize(self, weight_matrix):
        """
        量化整个权重矩阵
        
        Args:
            weight_matrix: 权重矩阵
            
        Returns:
            reconstructed: 重建后的矩阵
            block_info: 每个块的信息
        """
        rows, cols = weight_matrix.shape
        
        # 分析分块秩
        block_ranks, low_rank_blocks, high_rank_blocks = analyze_block_rank(
            weight_matrix,
            self.block_size,
            self.rank_threshold
        )
        
        print(f"分块分析:")
        print(f"  总块数: {len(block_ranks)}")
        print(f"  低秩块数: {len(low_rank_blocks)}")
        print(f"  满秩块数: {len(high_rank_blocks)}")
        print(f"  平均秩: {np.mean(block_ranks):.3f}")
        
        # 创建重建矩阵
        reconstructed = np.zeros_like(weight_matrix)
        block_info = []
        
        # 处理低秩块
        for i, j, start_row, end_row, start_col, end_col in low_rank_blocks:
            block = weight_matrix[start_row:end_row, start_col:end_col]
            
            # 量化
            quantized_block, num_levels, codebook_size = self.quantize_block(block, is_low_rank=True)
            
            # 放置到重建矩阵
            reconstructed[start_row:end_row, start_col:end_col] = quantized_block
            
            # 记录信息
            block_info.append({
                'type': 'low_rank',
                'position': (i, j),
                'range': (start_row, end_row, start_col, end_col),
                'num_levels': num_levels,
                'codebook_size': codebook_size,
                'rank': block_ranks[i * ((cols + self.block_size - 1) // self.block_size) + j]
            })
        
        # 处理满秩块
        for i, j, start_row, end_row, start_col, end_col in high_rank_blocks:
            block = weight_matrix[start_row:end_row, start_col:end_col]
            
            # 量化
            quantized_block, num_levels, codebook_size = self.quantize_block(block, is_low_rank=False)
            
            # 放置到重建矩阵
            reconstructed[start_row:end_row, start_col:end_col] = quantized_block
            
            # 记录信息
            block_info.append({
                'type': 'high_rank',
                'position': (i, j),
                'range': (start_row, end_row, start_col, end_col),
                'num_levels': num_levels,
                'codebook_size': codebook_size,
                'rank': block_ranks[i * ((cols + self.block_size - 1) // self.block_size) + j]
            })
        
        return reconstructed, block_info


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
                    expert_weight = weight[expert_idx].float().numpy()
                    print(f"选择专家 {expert_idx}, 形状: {expert_weight.shape}")
                    return expert_weight, key
                else:
                    return weight.float().numpy(), key
    
    return None, None


def test_hybrid_rvq():
    """
    测试混合粒度 RVQ
    """
    print("=" * 70)
    print("混合粒度 RVQ 测试")
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
    
    # 测试不同分块大小
    block_sizes = [32, 64, 128]
    
    results = {}
    
    for block_size in block_sizes:
        print(f"\n{'='*50}")
        print(f"分块大小: {block_size}")
        print(f"{'='*50}")
        
        # 创建混合 RVQ
        hybrid_rvq = HybridRVQ(block_size=block_size, rank_threshold=0.5)
        
        # 量化
        start_time = time.time()
        reconstructed, block_info = hybrid_rvq.quantize(expert_weight)
        end_time = time.time()
        
        # 计算误差
        error = np.linalg.norm(expert_weight - reconstructed) / np.linalg.norm(expert_weight)
        print(f"量化误差: {error*100:.2f}%")
        print(f"量化时间: {end_time - start_time:.2f}秒")
        
        # 统计块信息
        low_rank_blocks = [b for b in block_info if b['type'] == 'low_rank']
        high_rank_blocks = [b for b in block_info if b['type'] == 'high_rank']
        
        print(f"低秩块数: {len(low_rank_blocks)}")
        print(f"满秩块数: {len(high_rank_blocks)}")
        
        # 计算压缩比
        # 低秩块：2级，512码本
        # 满秩块：4级，128码本
        total_elements = expert_weight.size
        compressed_elements = 0
        
        for b in block_info:
            block_rows = b['range'][1] - b['range'][0]
            block_cols = b['range'][3] - b['range'][2]
            block_size_actual = block_rows * block_cols
            
            if b['type'] == 'low_rank':
                # 2级，每级需要 log2(512) = 9 位
                compressed_elements += block_size_actual * 2 * 9 / 32
            else:
                # 4级，每级需要 log2(128) = 7 位
                compressed_elements += block_size_actual * 4 * 7 / 32
        
        compression_ratio = total_elements / compressed_elements
        print(f"压缩比: {compression_ratio:.2f}x")
        
        results[block_size] = {
            'error': error,
            'compression_ratio': compression_ratio,
            'low_rank_blocks': len(low_rank_blocks),
            'high_rank_blocks': len(high_rank_blocks),
            'time': end_time - start_time
        }
    
    # 与标准 RVQ 对比
    print(f"\n{'='*50}")
    print(f"与标准 RVQ 对比")
    print(f"{'='*50}")
    
    # 标准 RVQ（4级，256码本）
    vectors = expert_weight.copy()
    residual = vectors.copy()
    rvq_quantized = []
    
    start_time = time.time()
    for level in range(4):
        actual_codebook_size = min(256, len(residual))
        kmeans = KMeans(n_clusters=actual_codebook_size, max_iter=50, random_state=42, n_init=10)
        kmeans.fit(residual)
        
        labels = kmeans.predict(residual)
        quantized = kmeans.cluster_centers_[labels]
        rvq_quantized.append(quantized)
        
        residual = residual - quantized
    
    rvq_reconstructed = sum(rvq_quantized)
    end_time = time.time()
    
    rvq_error = np.linalg.norm(expert_weight - rvq_reconstructed) / np.linalg.norm(expert_weight)
    print(f"标准 RVQ 误差: {rvq_error*100:.2f}%")
    print(f"标准 RVQ 时间: {end_time - start_time:.2f}秒")
    
    # 保存结果
    output_file = "hybrid_rvq_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("混合粒度 RVQ 测试结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试权重: " + selected_key + "\n")
        f.write("权重形状: " + str(expert_weight.shape) + "\n\n")
        
        f.write("不同分块大小结果:\n")
        for block_size, result in results.items():
            f.write(f"\n分块大小 {block_size}:\n")
            f.write(f"  量化误差: {result['error']*100:.2f}%\n")
            f.write(f"  压缩比: {result['compression_ratio']:.2f}x\n")
            f.write(f"  低秩块数: {result['low_rank_blocks']}\n")
            f.write(f"  满秩块数: {result['high_rank_blocks']}\n")
            f.write(f"  量化时间: {result['time']:.2f}秒\n")
        
        f.write(f"\n标准 RVQ 结果:\n")
        f.write(f"  量化误差: {rvq_error*100:.2f}%\n")
        f.write(f"  量化时间: {end_time - start_time:.2f}秒\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    # 运行测试
    results = test_hybrid_rvq()