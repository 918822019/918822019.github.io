"""
LoSparse vs 其他方法的全面对比

测试方法:
1. 纯SVD低秩分解
2. 纯稀疏化 (Magnitude)
3. LoSparse = SVD + Sparse
4. INT4量化
"""

import torch
import numpy as np
from safetensors import safe_open
import json
import time

model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"


def load_index():
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        return json.load(f)


def svd_compress(weight_np, rank):
    """纯SVD低秩压缩"""
    U, S, Vh = np.linalg.svd(weight_np, full_matrices=False)
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]
    reconstructed = U_k @ np.diag(S_k) @ Vh_k
    
    error = np.linalg.norm(weight_np - reconstructed) / np.linalg.norm(weight_np)
    
    # 存储量: U(m,r) + S(r) + Vh(r,n)
    storage = U_k.size + S_k.size + Vh_k.size
    compression = weight_np.size / storage
    
    return error, compression, "SVD"


def sparse_compress(weight_np, sparsity_ratio):
    """纯稀疏化压缩"""
    abs_weight = np.abs(weight_np)
    threshold = np.percentile(abs_weight.flatten(), (1 - sparsity_ratio) * 100)
    mask = abs_weight >= threshold
    sparse = weight_np * mask
    
    error = np.linalg.norm(weight_np - sparse) / np.linalg.norm(weight_np)
    
    # 存储量: 非零元素值 + 索引
    nnz = np.sum(mask)
    storage = nnz * 2  # 值 + 索引
    compression = weight_np.size / storage
    
    return error, compression, "Sparse"


def losparse_compress(weight_np, rank, sparsity_ratio):
    """LoSparse压缩: W = AB + S"""
    # Step 1: SVD低秩分解
    U, S, Vh = np.linalg.svd(weight_np, full_matrices=False)
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]
    lowrank = U_k @ np.diag(S_k) @ Vh_k
    
    # Step 2: 计算残差
    residual = weight_np - lowrank
    
    # Step 3: 稀疏化残差
    abs_residual = np.abs(residual)
    threshold = np.percentile(abs_residual.flatten(), (1 - sparsity_ratio) * 100)
    mask = abs_residual >= threshold
    sparse_residual = residual * mask
    
    # Step 4: 重构
    reconstructed = lowrank + sparse_residual
    
    error = np.linalg.norm(weight_np - reconstructed) / np.linalg.norm(weight_np)
    
    # 存储量: 低秩部分 + 稀疏部分
    lowrank_storage = U_k.size + S_k.size + Vh_k.size
    sparse_nnz = np.sum(mask)
    sparse_storage = sparse_nnz * 2
    total_storage = lowrank_storage + sparse_storage
    compression = weight_np.size / total_storage
    
    return error, compression, "LoSparse"


def int4_quantize(weight_np, group_size=128):
    """模拟INT4量化"""
    # 对称量化
    rows, cols = weight_np.shape
    
    # 分组
    num_groups = cols // group_size
    
    errors = []
    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        group = weight_np[:, start:end]
        
        # 计算scale
        max_val = np.max(np.abs(group))
        if max_val == 0:
            continue
        
        scale = max_val / 7.0  # INT4范围: -7 to 7
        
        # 量化
        quantized = np.clip(np.round(group / scale), -7, 7).astype(np.int8)
        
        # 反量化
        dequantized = quantized.astype(np.float32) * scale
        
        # 计算误差
        group_error = np.linalg.norm(group - dequantized) / np.linalg.norm(group)
        errors.append(group_error)
    
    avg_error = np.mean(errors)
    
    # 存储量: INT4值 + FP16 scale (每组一个)
    storage = (rows * cols) / 2 + num_groups * rows * 2  # 4bit值 + 16bit scale
    compression = (rows * cols * 4) / storage  # 原始FP32 -> 压缩后
    
    return avg_error, compression, "INT4"


def load_expert_weight(file_path, layer_pattern, expert_idx=0):
    """加载单个专家权重"""
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if layer_pattern in key:
                weight = f.get_tensor(key).float().numpy()
                if weight.ndim == 3:
                    return weight[expert_idx]
                return weight
    return None


def run_comparison(weight_np, layer_name):
    """运行所有方法的对比"""
    print(f"\n{'='*80}")
    print(f"层: {layer_name}")
    print(f"形状: {weight_np.shape}")
    print(f"{'='*80}")
    
    results = []
    
    # 1. 纯SVD
    for rank_ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        rank = int(min(weight_np.shape) * rank_ratio)
        error, comp, method = svd_compress(weight_np, rank)
        results.append({
            "method": method,
            "rank_ratio": rank_ratio,
            "rank": rank,
            "sparsity": 0,
            "error": error,
            "compression": comp
        })
    
    # 2. 纯稀疏化
    for sparsity in [0.05, 0.10, 0.15, 0.20, 0.30]:
        error, comp, method = sparse_compress(weight_np, sparsity)
        results.append({
            "method": method,
            "rank_ratio": 0,
            "rank": 0,
            "sparsity": sparsity,
            "error": error,
            "compression": comp
        })
    
    # 3. LoSparse
    for rank_ratio in [0.2, 0.3, 0.4, 0.5]:
        for sparsity in [0.05, 0.10, 0.15, 0.20]:
            rank = int(min(weight_np.shape) * rank_ratio)
            error, comp, method = losparse_compress(weight_np, rank, sparsity)
            results.append({
                "method": method,
                "rank_ratio": rank_ratio,
                "rank": rank,
                "sparsity": sparsity,
                "error": error,
                "compression": comp
            })
    
    # 4. INT4量化
    error, comp, method = int4_quantize(weight_np)
    results.append({
        "method": method,
        "rank_ratio": 0,
        "rank": 0,
        "sparsity": 0,
        "error": error,
        "compression": comp
    })
    
    return results


def print_results_table(results, title):
    """打印结果表格"""
    print(f"\n{title}")
    print(f"{'Method':<12} {'Rank%':<8} {'Sparsity%':<12} {'Error%':<12} {'Compression':<15}")
    print("-" * 60)
    
    # 按误差排序
    sorted_results = sorted(results, key=lambda x: x["error"])
    
    for r in sorted_results[:15]:  # 只显示前15个
        rank_str = f"{r['rank_ratio']*100:.0f}" if r['rank_ratio'] > 0 else "-"
        sparsity_str = f"{r['sparsity']*100:.0f}" if r['sparsity'] > 0 else "-"
        
        print(f"{r['method']:<12} {rank_str:<8} {sparsity_str:<12} "
              f"{r['error']*100:<12.2f} {r['compression']:<15.2f}")


def find_best_configs(results, error_threshold):
    """找最优配置"""
    filtered = [r for r in results if r["error"] < error_threshold]
    if not filtered:
        return None
    
    best = max(filtered, key=lambda x: x["compression"])
    return best


def main():
    print("=" * 80)
    print("LoSparse vs 其他方法全面对比")
    print("=" * 80)
    
    # 加载索引
    index_data = load_index()
    weight_map = index_data["weight_map"]
    
    # 找MoE专家层
    moe_layers = []
    for name in weight_map.keys():
        if "mlp.experts.gate_up_proj" in name or "mlp.experts.down_proj" in name:
            moe_layers.append(name)
    
    # 选择几个层进行测试
    test_layers = moe_layers[:2]  # 测试前2个层
    
    all_results = []
    
    for layer_name in test_layers:
        file_path = f"{model_path}/{weight_map[layer_name]}"
        
        # 加载第一个专家
        weight = load_expert_weight(file_path, layer_name, expert_idx=0)
        
        if weight is None:
            continue
        
        results = run_comparison(weight, layer_name)
        all_results.extend(results)
        
        # 打印该层的结果
        print_results_table(results, f"层 {layer_name.split('.')[-2]} 的结果")
    
    # 汇总所有结果
    print("\n" + "=" * 80)
    print("汇总: 所有层的结果")
    print("=" * 80)
    
    # 按方法分组
    method_results = {}
    for r in all_results:
        method = r["method"]
        if method not in method_results:
            method_results[method] = []
        method_results[method].append(r)
    
    # 找每种方法的最优配置
    print("\n各方法最优配置 (误差<20%):")
    print(f"{'Method':<12} {'Rank%':<8} {'Sparsity%':<12} {'Error%':<12} {'Compression':<15}")
    print("-" * 60)
    
    for method in ["SVD", "Sparse", "LoSparse", "INT4"]:
        if method in method_results:
            best = find_best_configs(method_results[method], 0.20)
            if best:
                rank_str = f"{best['rank_ratio']*100:.0f}" if best['rank_ratio'] > 0 else "-"
                sparsity_str = f"{best['sparsity']*100:.0f}" if best['sparsity'] > 0 else "-"
                print(f"{method:<12} {rank_str:<8} {sparsity_str:<12} "
                      f"{best['error']*100:<12.2f} {best['compression']:<15.2f}")
    
    # 找误差<10%的配置
    print("\n各方法最优配置 (误差<10%):")
    print(f"{'Method':<12} {'Rank%':<8} {'Sparsity%':<12} {'Error%':<12} {'Compression':<15}")
    print("-" * 60)
    
    for method in ["SVD", "Sparse", "LoSparse", "INT4"]:
        if method in method_results:
            best = find_best_configs(method_results[method], 0.10)
            if best:
                rank_str = f"{best['rank_ratio']*100:.0f}" if best['rank_ratio'] > 0 else "-"
                sparsity_str = f"{best['sparsity']*100:.0f}" if best['sparsity'] > 0 else "-"
                print(f"{method:<12} {rank_str:<8} {sparsity_str:<12} "
                      f"{best['error']*100:<12.2f} {best['compression']:<15.2f}")
            else:
                print(f"{method:<12} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
    
    # 关键发现
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)
    
    # 分析LoSparse vs 纯SVD
    svd_errors = [r["error"] for r in all_results if r["method"] == "SVD"]
    losparse_errors = [r["error"] for r in all_results if r["method"] == "LoSparse"]
    
    if svd_errors and losparse_errors:
        svd_avg = np.mean(svd_errors)
        losparse_avg = np.mean(losparse_errors)
        improvement = (svd_avg - losparse_avg) / svd_avg * 100
        
        print(f"\n1. LoSparse vs 纯SVD:")
        print(f"   平均误差: SVD={svd_avg*100:.1f}%, LoSparse={losparse_avg*100:.1f}%")
        print(f"   改进: {improvement:.1f}%")
        
        if improvement > 10:
            print("   ✅ LoSparse明显优于纯SVD")
        elif improvement > 0:
            print("   ⚠️  LoSparse略优于纯SVD")
        else:
            print("   ❌ LoSparse不优于纯SVD")


if __name__ == "__main__":
    main()