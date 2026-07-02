"""
仅测试 RVQ 量化，不加载模型
"""

import torch
import numpy as np
from safetensors import safe_open
import json
import time
from sklearn.cluster import KMeans

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"

def load_index():
    """加载模型索引"""
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        return json.load(f)

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

def rvq_quantize(weight_matrix, num_levels=4, codebook_size=256):
    """
    对权重矩阵进行 RVQ 量化
    """
    vectors = weight_matrix.copy()
    residual = vectors.copy()
    all_quantized = []
    
    for level in range(num_levels):
        # 确保码本大小不超过向量数量
        actual_codebook_size = min(codebook_size, len(residual))
        
        kmeans = KMeans(n_clusters=actual_codebook_size, max_iter=50, random_state=42, n_init=10)
        kmeans.fit(residual)
        
        labels = kmeans.predict(residual)
        quantized = kmeans.cluster_centers_[labels]
        all_quantized.append(quantized)
        
        residual = residual - quantized
    
    # 重建到各级
    reconstructions = []
    for level in range(num_levels):
        recon = np.sum(all_quantized[:level+1], axis=0)
        reconstructions.append(recon)
    
    return reconstructions, all_quantized

def analyze_rvq_semantic_layering(weight_matrix, num_levels=4, codebook_size=256):
    """
    分析 RVQ 的语义分层特性
    """
    print("=" * 70)
    print("RVQ 语义分层分析")
    print("=" * 70)
    
    # RVQ 量化
    reconstructions, all_quantized = rvq_quantize(weight_matrix, num_levels, codebook_size)
    
    level_names = ["L1 (任务相关)", "L2 (句法结构)", "L3 (知识细节)", "L4 (残差噪声)"]
    
    # 计算各级重建误差
    print("\n各级重建误差:")
    for i, recon in enumerate(reconstructions):
        error = np.linalg.norm(weight_matrix - recon) / np.linalg.norm(weight_matrix)
        print(f"  {level_names[i]}: {error*100:.2f}%")
    
    # 计算各级能量
    print("\n各级能量:")
    total_energy = 0
    level_energies = []
    for i, quantized in enumerate(all_quantized):
        energy = np.linalg.norm(quantized)**2
        level_energies.append(energy)
        total_energy += energy
        print(f"  {level_names[i]}: {energy:.2f}")
    
    # 计算各级能量比例
    print("\n各级能量比例:")
    for i, energy in enumerate(level_energies):
        ratio = energy / total_energy
        print(f"  {level_names[i]}: {ratio*100:.2f}%")
    
    # 分析各级量化结果的统计特性
    print("\n各级量化结果统计特性:")
    for i, quantized in enumerate(all_quantized):
        mean_val = np.mean(quantized)
        std_val = np.std(quantized)
        norm_val = np.linalg.norm(quantized)
        print(f"  {level_names[i]}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 范数={norm_val:.2f}")
    
    # 分析各级残差的统计特性
    print("\n各级残差统计特性:")
    residual = weight_matrix.copy()
    for i in range(num_levels):
        quantized = all_quantized[i]
        residual = residual - quantized
        
        mean_val = np.mean(residual)
        std_val = np.std(residual)
        norm_val = np.linalg.norm(residual)
        print(f"  L{i+1} 残差: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 范数={norm_val:.2f}")
    
    # 分析各级码本的差异性
    print("\n各级码本差异性分析:")
    codebooks = []
    residual = weight_matrix.copy()
    for i in range(num_levels):
        actual_codebook_size = min(codebook_size, len(residual))
        kmeans = KMeans(n_clusters=actual_codebook_size, max_iter=50, random_state=42, n_init=10)
        kmeans.fit(residual)
        codebooks.append(kmeans.cluster_centers_)
        
        labels = kmeans.predict(residual)
        quantized = kmeans.cluster_centers_[labels]
        residual = residual - quantized
    
    for i in range(num_levels-1):
        codebook1 = codebooks[i]
        codebook2 = codebooks[i+1]
        
        # 计算码本中心之间的距离矩阵
        distances = np.linalg.norm(
            codebook1[:, np.newaxis, :] - codebook2[np.newaxis, :, :], 
            axis=2
        )
        
        # 平均最近距离
        min_distances = np.min(distances, axis=1)
        avg_distance = np.mean(min_distances)
        print(f"  {level_names[i]} vs {level_names[i+1]}: 平均距离={avg_distance:.4f}")
    
    # 语义分层假设验证
    print("\n" + "=" * 70)
    print("语义分层假设验证")
    print("=" * 70)
    
    # 假设：不同级别可能捕捉不同语义层次
    # L1: 任务相关信息 (高频、大尺度模式)
    # L2: 句法信息 (中频、结构模式)
    # L3: 知识信息 (低频、细节模式)
    # L4: 残差噪声
    
    # 分析各级能量比例
    print("\n基于能量比例的语义分层分析:")
    for i, energy in enumerate(level_energies):
        ratio = energy / total_energy
        if ratio > 0.3:
            print(f"  {level_names[i]}: 高能量 ({ratio*100:.1f}%) - 可能捕捉主要模式")
        elif ratio > 0.2:
            print(f"  {level_names[i]}: 中等能量 ({ratio*100:.1f}%) - 可能捕捉结构模式")
        else:
            print(f"  {level_names[i]}: 低能量 ({ratio*100:.1f}%) - 可能捕捉细节或噪声")
    
    # 分析各级重建误差的下降速度
    print("\n基于重建误差下降速度的分析:")
    errors = []
    for i, recon in enumerate(reconstructions):
        error = np.linalg.norm(weight_matrix - recon) / np.linalg.norm(weight_matrix)
        errors.append(error)
    
    for i in range(1, len(errors)):
        error_reduction = errors[i-1] - errors[i]
        print(f"  L{i} -> L{i+1}: 误差减少 {error_reduction*100:.2f}%")
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    print("\n1. RVQ 能够将权重矩阵分解为多个级别")
    print("2. 每个级别捕捉不同尺度的信息")
    print("3. 语义分层假设需要进一步验证（需要 Probe Model）")
    
    return {
        'reconstructions': reconstructions,
        'all_quantized': all_quantized,
        'level_energies': level_energies,
        'errors': errors
    }

def test_single_expert():
    """
    测试单个 Expert 的 RVQ 分层特性
    """
    print("=" * 70)
    print("测试 Qwen3.5-35B-A3B 单个 Expert 的 RVQ 分层特性")
    print("=" * 70)
    
    # 加载模型索引
    index_data = load_index()
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
    
    # 分析 RVQ 分层特性
    results = analyze_rvq_semantic_layering(expert_weight, num_levels=4, codebook_size=256)
    
    # 测试不同码本大小
    print("\n" + "=" * 70)
    print("测试不同码本大小的影响")
    print("=" * 70)
    
    codebook_sizes = [64, 128, 256, 512]
    for codebook_size in codebook_sizes:
        print(f"\n码本大小: {codebook_size}")
        print("-" * 30)
        
        # 只测试前两级
        reconstructions, _ = rvq_quantize(expert_weight, num_levels=2, codebook_size=codebook_size)
        
        for i, recon in enumerate(reconstructions):
            error = np.linalg.norm(expert_weight - recon) / np.linalg.norm(expert_weight)
            print(f"  L{i+1} 重建误差: {error*100:.2f}%")
    
    # 保存结果
    output_file = "rvq_semantic_layering_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Qwen3.5-35B-A3B Expert RVQ 语义分层分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试权重: " + selected_key + "\n")
        f.write("权重形状: " + str(expert_weight.shape) + "\n\n")
        
        f.write("各级重建误差:\n")
        for i, recon in enumerate(results['reconstructions']):
            error = np.linalg.norm(expert_weight - recon) / np.linalg.norm(expert_weight)
            f.write(f"  L{i+1}: {error*100:.2f}%\n")
        
        f.write("\n各级能量比例:\n")
        total_energy = sum(results['level_energies'])
        for i, energy in enumerate(results['level_energies']):
            ratio = energy / total_energy
            f.write(f"  L{i+1}: {ratio*100:.2f}%\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    # 运行测试
    results = test_single_expert()