"""
LoSparse / DSL (Decomposition with Sparse and Low-rank) 实现

核心思想: W = AB + S
- AB: 低秩部分 (用SVD初始化)
- S: 稀疏残差部分 (用Magnitude/Wanda方法保留重要元素)

优势:
1. 免训练，快速验证
2. 结合低秩和稀疏性的互补优势
3. 适合MoE专家层压缩
"""

import torch
import numpy as np
from safetensors import safe_open
from collections import defaultdict
import json
import gc
import time

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"


def load_index():
    """加载模型索引"""
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        return json.load(f)


def svd_lowrank(weight_np, rank):
    """
    SVD低秩分解
    返回: U, S, Vh (取前rank个奇异值)
    """
    U, S, Vh = np.linalg.svd(weight_np, full_matrices=False)
    
    # 取前rank个
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]
    
    return U_k, S_k, Vh_k


def reconstruct_lowrank(U_k, S_k, Vh_k):
    """从低秩分量重构矩阵"""
    # AB = U_k @ diag(S_k) @ Vh_k
    return U_k @ np.diag(S_k) @ Vh_k


def magnitude_sparsify(weight_np, sparsity_ratio):
    """
    Magnitude稀疏化: 保留绝对值最大的元素
    
    参数:
        weight_np: 输入矩阵
        sparsity_ratio: 保留比例 (0.05 = 保留5%)
    
    返回:
        sparse_matrix: 稀疏矩阵 (大部分为0)
        mask: 保留位置的mask
    """
    # 计算绝对值
    abs_weight = np.abs(weight_np)
    
    # 计算阈值
    threshold = np.percentile(abs_weight.flatten(), (1 - sparsity_ratio) * 100)
    
    # 创建mask
    mask = abs_weight >= threshold
    
    # 应用mask
    sparse_matrix = weight_np * mask
    
    return sparse_matrix, mask


def wanda_sparsify(weight_np, activation_np, sparsity_ratio):
    """
    Wanda稀疏化: 基于权重和激活值的联合重要性
    
    Wanda评分 = |W| * ||X||_2 (按列的L2范数)
    
    参数:
        weight_np: 权重矩阵 (out_features, in_features)
        activation_np: 激活值 (batch, in_features) 或 None
        sparsity_ratio: 保留比例
    
    返回:
        sparse_matrix: 稀疏矩阵
        mask: 保留位置的mask
    """
    if activation_np is None:
        # 如果没有激活值，退化为Magnitude方法
        return magnitude_sparsify(weight_np, sparsity_ratio)
    
    # 计算激活值的L2范数 (按列)
    act_norm = np.sqrt(np.sum(activation_np ** 2, axis=0))  # (in_features,)
    
    # 计算Wanda评分
    wanda_score = np.abs(weight_np) * act_norm[np.newaxis, :]  # (out, in)
    
    # 计算阈值
    threshold = np.percentile(wanda_score.flatten(), (1 - sparsity_ratio) * 100)
    
    # 创建mask
    mask = wanda_score >= threshold
    
    # 应用mask
    sparse_matrix = weight_np * mask
    
    return sparse_matrix, mask


def losparse_compress(weight_np, rank, sparsity_ratio, method="magnitude"):
    """
    LoSparse压缩: W = AB + S
    
    参数:
        weight_np: 原始权重矩阵
        rank: 低秩分解的秩
        sparsity_ratio: 稀疏残差保留比例
        method: 稀疏化方法 ("magnitude" 或 "wanda")
    
    返回:
        result: 压缩后的结果字典
    """
    start_time = time.time()
    
    # Step 1: SVD低秩分解
    U_k, S_k, Vh_k = svd_lowrank(weight_np, rank)
    lowrank_recon = reconstruct_lowrank(U_k, S_k, Vh_k)
    
    # Step 2: 计算残差
    residual = weight_np - lowrank_recon
    
    # Step 3: 稀疏化残差
    if method == "magnitude":
        sparse_residual, mask = magnitude_sparsify(residual, sparsity_ratio)
    elif method == "wanda":
        sparse_residual, mask = wanda_sparsify(residual, None, sparsity_ratio)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 4: 重构
    reconstructed = lowrank_recon + sparse_residual
    
    # 计算误差
    error = np.linalg.norm(weight_np - reconstructed) / np.linalg.norm(weight_np)
    
    # 计算存储量
    # 低秩部分: U_k (m, rank) + S_k (rank,) + Vh_k (rank, n)
    lowrank_params = U_k.size + S_k.size + Vh_k.size
    # 稀疏部分: 非零元素个数 + 索引开销
    sparse_nnz = np.sum(mask)
    sparse_params = sparse_nnz * 2  # 值 + 索引
    total_params = lowrank_params + sparse_params
    original_params = weight_np.size
    compression_ratio = original_params / total_params
    
    elapsed = time.time() - start_time
    
    return {
        "U_k": U_k,
        "S_k": S_k,
        "Vh_k": Vh_k,
        "sparse_residual": sparse_residual,
        "mask": mask,
        "reconstructed": reconstructed,
        "error": error,
        "compression_ratio": compression_ratio,
        "lowrank_params": lowrank_params,
        "sparse_params": sparse_params,
        "total_params": total_params,
        "original_params": original_params,
        "elapsed": elapsed
    }


def analyze_layer_losparse(weight_np, layer_name, rank_ratios=[0.2, 0.3, 0.4], 
                           sparsity_ratios=[0.05, 0.10, 0.15]):
    """
    分析单层的LoSparse效果
    """
    print(f"\n{'='*70}")
    print(f"分析层: {layer_name}")
    print(f"形状: {weight_np.shape}")
    print(f"{'='*70}")
    
    results = []
    
    for rank_ratio in rank_ratios:
        for sparsity_ratio in sparsity_ratios:
            rank = int(min(weight_np.shape) * rank_ratio)
            
            result = losparse_compress(weight_np, rank, sparsity_ratio, method="magnitude")
            
            results.append({
                "rank_ratio": rank_ratio,
                "rank": rank,
                "sparsity_ratio": sparsity_ratio,
                "error": result["error"],
                "compression_ratio": result["compression_ratio"],
                "elapsed": result["elapsed"]
            })
            
            print(f"  rank={rank} ({rank_ratio*100:.0f}%), sparsity={sparsity_ratio*100:.0f}%: "
                  f"error={result['error']*100:.2f}%, compression={result['compression_ratio']:.2f}x")
    
    return results


def load_expert_weights(file_path, layer_name_pattern, max_experts=5):
    """加载MoE专家权重"""
    results = []
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            # 找到目标层
            target_keys = [k for k in keys if layer_name_pattern in k]
            
            if not target_keys:
                return results
            
            for key in target_keys[:1]:  # 只取第一个匹配的层
                weight = f.get_tensor(key)
                weight_np = weight.float().numpy()
                
                if weight_np.ndim == 3:
                    # MoE专家层: (num_experts, out_features, in_features)
                    num_experts = min(weight_np.shape[0], max_experts)
                    for i in range(num_experts):
                        results.append({
                            "name": f"{key}.expert_{i}",
                            "weight": weight_np[i]
                        })
                elif weight_np.ndim == 2:
                    results.append({
                        "name": key,
                        "weight": weight_np
                    })
                    
    except Exception as e:
        print(f"加载文件出错: {e}")
    
    return results


def main():
    print("=" * 70)
    print("LoSparse / DSL 实现与测试")
    print("=" * 70)
    
    # 加载索引
    index_data = load_index()
    weight_map = index_data["weight_map"]
    
    # 找到MoE专家层
    moe_layers = []
    for name in weight_map.keys():
        if "mlp.experts.gate_up_proj" in name or "mlp.experts.down_proj" in name:
            moe_layers.append(name)
    
    print(f"\n找到 {len(moe_layers)} 个MoE专家层")
    
    # 选择一个层进行详细分析
    target_layer = moe_layers[0]
    target_file = weight_map[target_layer]
    file_path = f"{model_path}/{target_file}"
    
    print(f"\n选择层: {target_layer}")
    print(f"文件: {target_file}")
    
    # 加载权重
    experts = load_expert_weights(file_path, target_layer, max_experts=3)
    
    if not experts:
        print("未能加载权重")
        return
    
    # 测试不同配置
    print("\n" + "=" * 70)
    print("测试LoSparse不同配置")
    print("=" * 70)
    
    all_results = []
    
    for expert in experts:
        print(f"\n{'='*70}")
        print(f"专家: {expert['name']}")
        print(f"形状: {expert['weight'].shape}")
        print(f"{'='*70}")
        
        weight = expert["weight"]
        
        # 测试不同rank和sparsity组合
        results = analyze_layer_losparse(
            weight, 
            expert["name"],
            rank_ratios=[0.2, 0.3, 0.4, 0.5],
            sparsity_ratios=[0.03, 0.05, 0.10, 0.15, 0.20]
        )
        
        all_results.extend(results)
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    
    # 按压缩比排序
    all_results.sort(key=lambda x: x["compression_ratio"], reverse=True)
    
    print(f"\n{'Rank%':<8} {'Sparsity%':<12} {'Error%':<12} {'Compression':<15}")
    print("-" * 50)
    
    for r in all_results[:20]:  # 只显示前20个
        print(f"{r['rank_ratio']*100:<8.0f} {r['sparsity_ratio']*100:<12.0f} "
              f"{r['error']*100:<12.2f} {r['compression_ratio']:<15.2f}")
    
    # 找到最优配置
    print("\n" + "=" * 70)
    print("最优配置推荐")
    print("=" * 70)
    
    # 按误差过滤，找最高压缩比
    low_error = [r for r in all_results if r["error"] < 0.10]  # 误差<10%
    if low_error:
        best = max(low_error, key=lambda x: x["compression_ratio"])
        print(f"\n误差<10%的最优配置:")
        print(f"  Rank: {best['rank']} ({best['rank_ratio']*100:.0f}%)")
        print(f"  Sparsity: {best['sparsity_ratio']*100:.0f}%")
        print(f"  Error: {best['error']*100:.2f}%")
        print(f"  Compression: {best['compression_ratio']:.2f}x")
    
    medium_error = [r for r in all_results if r["error"] < 0.20]  # 误差<20%
    if medium_error:
        best = max(medium_error, key=lambda x: x["compression_ratio"])
        print(f"\n误差<20%的最优配置:")
        print(f"  Rank: {best['rank']} ({best['rank_ratio']*100:.0f}%)")
        print(f"  Sparsity: {best['sparsity_ratio']*100:.0f}%")
        print(f"  Error: {best['error']*100:.2f}%")
        print(f"  Compression: {best['compression_ratio']:.2f}x")


if __name__ == "__main__":
    main()