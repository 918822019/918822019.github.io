"""
验证语义分层假设：对Qwen3.5-35B-A3B的单个Expert进行4级RVQ
然后用Probe Model分析各级码本重建后的隐藏状态

RVQ (Residual Vector Quantization) 原理：
1. 第一级：用码本C1量化原始向量x，得到q1，残差r1 = x - q1
2. 第二级：用码本C2量化残差r1，得到q2，残差r2 = r1 - q2
3. 第三级：用码本C3量化残差r2，得到q3，残差r3 = r2 - q3
4. 第四级：用码本C4量化残差r3，得到q4，残差r4 = r3 - q4
最终量化结果：x_quantized = q1 + q2 + q3 + q4
"""

import torch
import numpy as np
from safetensors import safe_open
from collections import defaultdict
import json
import gc
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"

def load_index():
    """加载模型索引"""
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        return json.load(f)

def load_expert_weight(file_path, expert_name, expert_idx=0):
    """
    加载指定Expert的权重
    对于MoE层，权重是3D的: (num_experts, out_dim, in_dim)
    """
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        # 查找目标权重
        for key in keys:
            if expert_name in key:
                weight = f.get_tensor(key)
                print(f"加载权重: {key}, 形状: {weight.shape}")
                
                # 如果是3D张量，选择指定专家
                if weight.ndim == 3:
                    expert_weight = weight[expert_idx].float().numpy()
                    print(f"选择专家 {expert_idx}, 形状: {expert_weight.shape}")
                    return expert_weight
                else:
                    return weight.float().numpy()
    
    return None

class RVQQuantizer:
    """
    残差向量量化器 (Residual Vector Quantizer)
    """
    def __init__(self, num_levels=4, codebook_size=256, max_iter=50):
        """
        Args:
            num_levels: RVQ级数
            codebook_size: 每级码本大小
            max_iter: K-means最大迭代次数
        """
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.max_iter = max_iter
        self.codebooks = []  # 存储各级码本
        self.cluster_centers = []  # 存储聚类中心
        
    def fit_codebooks(self, vectors):
        """
        训练各级码本
        
        Args:
            vectors: 形状为 (n_vectors, vector_dim) 的向量集合
        """
        print(f"训练 {self.num_levels} 级RVQ码本，每级 {self.codebook_size} 个码字")
        
        residual = vectors.copy()
        
        for level in range(self.num_levels):
            print(f"\n训练第 {level+1} 级码本...")
            
            # 确保码本大小不超过向量数量
            actual_codebook_size = min(self.codebook_size, len(residual))
            
            # 使用K-means聚类
            kmeans = KMeans(
                n_clusters=actual_codebook_size,
                max_iter=self.max_iter,
                random_state=42,
                n_init=10
            )
            kmeans.fit(residual)
            
            # 存储码本（聚类中心）
            self.codebooks.append(kmeans.cluster_centers_)
            self.cluster_centers.append(kmeans.cluster_centers_)
            
            # 计算量化结果和残差
            labels = kmeans.predict(residual)
            quantized = kmeans.cluster_centers_[labels]
            residual = residual - quantized
            
            # 计算误差
            error = np.linalg.norm(residual) / np.linalg.norm(vectors)
            print(f"  第 {level+1} 级残差相对误差: {error*100:.2f}%")
            print(f"  码本形状: {kmeans.cluster_centers_.shape}")
            
        print(f"\n码本训练完成！")
        
    def quantize(self, vectors):
        """
        使用训练好的码本对向量进行量化
        
        Args:
            vectors: 形状为 (n_vectors, vector_dim) 的向量集合
            
        Returns:
            quantized: 量化后的向量
            codes: 各级的编码索引
            residuals: 各级的残差
        """
        if not self.codebooks:
            raise ValueError("请先调用 fit_codebooks() 训练码本")
            
        residual = vectors.copy()
        all_codes = []
        all_quantized = []
        
        for level in range(self.num_levels):
            codebook = self.codebooks[level]
            
            # 找到最近的码字
            distances = np.linalg.norm(
                residual[:, np.newaxis, :] - codebook[np.newaxis, :, :], 
                axis=2
            )  # (n_vectors, codebook_size)
            
            codes = np.argmin(distances, axis=1)  # (n_vectors,)
            quantized = codebook[codes]  # (n_vectors, vector_dim)
            
            all_codes.append(codes)
            all_quantized.append(quantized)
            
            # 更新残差
            residual = residual - quantized
            
        # 重建的向量是各级量化结果的和
        reconstructed = np.sum(all_quantized, axis=0)
        
        return reconstructed, all_codes, all_quantized, residual
    
    def reconstruct_level(self, vectors, up_to_level):
        """
        重建到指定级别
        
        Args:
            vectors: 原始向量
            up_to_level: 重建到第几级 (1-based)
            
        Returns:
            reconstructed: 重建后的向量
        """
        if not self.codebooks:
            raise ValueError("请先调用 fit_codebooks() 训练码本")
            
        residual = vectors.copy()
        reconstructed = np.zeros_like(vectors)
        
        for level in range(min(up_to_level, self.num_levels)):
            codebook = self.codebooks[level]
            
            # 找到最近的码字
            distances = np.linalg.norm(
                residual[:, np.newaxis, :] - codebook[np.newaxis, :, :], 
                axis=2
            )
            
            codes = np.argmin(distances, axis=1)
            quantized = codebook[codes]
            
            reconstructed += quantized
            residual = residual - quantized
            
        return reconstructed

def analyze_rvq_levels(weight_matrix, num_levels=4, codebook_size=256):
    """
    分析权重矩阵的RVQ分层特性
    
    Args:
        weight_matrix: 2D权重矩阵 (n_rows, n_cols)
        num_levels: RVQ级数
        codebook_size: 每级码本大小
        
    Returns:
        analysis_results: 分析结果字典
    """
    print("=" * 70)
    print("开始RVQ分层分析")
    print("=" * 70)
    
    # 将权重矩阵转换为向量集合 (每行作为一个向量)
    vectors = weight_matrix.copy()
    print(f"向量集合形状: {vectors.shape}")
    print(f"向量数量: {vectors.shape[0]}, 向量维度: {vectors.shape[1]}")
    
    # 创建RVQ量化器
    rvq = RVQQuantizer(
        num_levels=num_levels,
        codebook_size=codebook_size,
        max_iter=100
    )
    
    # 训练码本
    rvq.fit_codebooks(vectors)
    
    # 量化向量
    print("\n进行RVQ量化...")
    reconstructed, all_codes, all_quantized, final_residual = rvq.quantize(vectors)
    
    # 计算各级误差
    errors = []
    for level in range(num_levels):
        # 重建到第level+1级
        recon_level = rvq.reconstruct_level(vectors, level+1)
        error = np.linalg.norm(vectors - recon_level) / np.linalg.norm(vectors)
        errors.append(error)
        print(f"第 {level+1} 级重建误差: {error*100:.2f}%")
    
    # 计算各级量化结果的能量
    level_energies = []
    for level in range(num_levels):
        energy = np.linalg.norm(all_quantized[level])**2
        level_energies.append(energy)
        print(f"第 {level+1} 级能量: {energy:.2f}")
    
    # 计算各级码本的统计特性
    codebook_stats = []
    for level in range(num_levels):
        codebook = rvq.codebooks[level]
        stats = {
            'mean': np.mean(codebook, axis=0),
            'std': np.std(codebook, axis=0),
            'norm': np.linalg.norm(codebook, axis=1)
        }
        codebook_stats.append(stats)
        
    # 计算各级残差的统计特性
    residual_stats = []
    residual = vectors.copy()
    for level in range(num_levels):
        codebook = rvq.codebooks[level]
        distances = np.linalg.norm(
            residual[:, np.newaxis, :] - codebook[np.newaxis, :, :], 
            axis=2
        )
        codes = np.argmin(distances, axis=1)
        quantized = codebook[codes]
        residual = residual - quantized
        
        stats = {
            'mean': np.mean(residual),
            'std': np.std(residual),
            'norm': np.linalg.norm(residual)
        }
        residual_stats.append(stats)
        
    # 分析各级码本的差异性
    print("\n分析各级码本的差异性...")
    codebook_differences = []
    for level in range(num_levels-1):
        # 计算相邻两级码本的平均距离
        codebook1 = rvq.codebooks[level]
        codebook2 = rvq.codebooks[level+1]
        
        # 计算码本中心之间的距离矩阵
        distances = np.linalg.norm(
            codebook1[:, np.newaxis, :] - codebook2[np.newaxis, :, :], 
            axis=2
        )
        
        # 平均最近距离
        min_distances = np.min(distances, axis=1)
        avg_distance = np.mean(min_distances)
        codebook_differences.append(avg_distance)
        print(f"第 {level+1} 级与第 {level+2} 级码本平均距离: {avg_distance:.4f}")
    
    # 语义分层分析假设
    print("\n" + "=" * 70)
    print("语义分层假设分析")
    print("=" * 70)
    
    # 假设：不同级别可能捕捉不同语义层次
    # L1: 任务相关信息 (高频、大尺度模式)
    # L2: 句法信息 (中频、结构模式)
    # L3: 知识信息 (低频、细节模式)
    # L4: 残差噪声
    
    # 分析各级能量比例
    total_energy = sum(level_energies)
    energy_ratios = [e/total_energy for e in level_energies]
    
    print("\n各级能量比例:")
    for i, ratio in enumerate(energy_ratios):
        level_name = f"L{i+1}"
        if i == 0:
            semantic = "任务相关"
        elif i == 1:
            semantic = "句法结构"
        elif i == 2:
            semantic = "知识细节"
        else:
            semantic = "残差噪声"
            
        print(f"  {level_name} ({semantic}): {ratio*100:.2f}%")
    
    # 分析各级量化误差的分布
    print("\n各级量化误差分布:")
    for level in range(num_levels):
        # 计算第level+1级的量化误差向量
        recon_level = rvq.reconstruct_level(vectors, level+1)
        error_vectors = vectors - recon_level
        
        # 计算误差向量的统计特性
        error_norms = np.linalg.norm(error_vectors, axis=1)
        print(f"  L{level+1}: 平均误差范数={np.mean(error_norms):.4f}, "
              f"标准差={np.std(error_norms):.4f}")
    
    return {
        'rvq': rvq,
        'vectors': vectors,
        'reconstructed': reconstructed,
        'all_codes': all_codes,
        'all_quantized': all_quantized,
        'final_residual': final_residual,
        'errors': errors,
        'level_energies': level_energies,
        'energy_ratios': energy_ratios,
        'codebook_stats': codebook_stats,
        'residual_stats': residual_stats,
        'codebook_differences': codebook_differences
    }

def pca_whitening(weight_matrix, n_components=None):
    """
    PCA白化预处理
    
    Args:
        weight_matrix: 输入权重矩阵
        n_components: 保留的主成分数量
        
    Returns:
        whitened: 白化后的矩阵
        pca: PCA对象
        scaler: 标准化器
    """
    print("进行PCA白化预处理...")
    
    # 标准化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(weight_matrix)
    
    # PCA
    if n_components is None:
        n_components = min(weight_matrix.shape)
        
    pca = PCA(n_components=n_components)
    whitened = pca.fit_transform(scaled)
    
    print(f"原始形状: {weight_matrix.shape}")
    print(f"白化后形状: {whitened.shape}")
    print(f"保留方差比例: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    return whitened, pca, scaler

def test_expert_rvq():
    """
    测试单个Expert的RVQ分层特性
    """
    print("=" * 70)
    print("Qwen3.5-35B-A3B Expert RVQ语义分层验证")
    print("=" * 70)
    
    # 加载模型索引
    index_data = load_index()
    weight_map = index_data["weight_map"]
    
    # 选择一个MoE专家层进行分析
    # 找到gate_up_proj或down_proj
    moe_keys = [k for k in weight_map.keys() if "experts" in k]
    
    if not moe_keys:
        print("未找到MoE专家层，尝试查找其他层...")
        moe_keys = [k for k in weight_map.keys() if "gate" in k or "up" in k or "down" in k]
    
    print(f"找到 {len(moe_keys)} 个MoE相关权重")
    
    # 选择第一个MoE层
    selected_key = moe_keys[0]
    print(f"\n选择权重: {selected_key}")
    
    # 找到对应的文件
    file_name = weight_map[selected_key]
    file_path = f"{model_path}/{file_name}"
    
    # 加载权重
    with safe_open(file_path, framework="pt", device="cpu") as f:
        weight = f.get_tensor(selected_key)
        
    print(f"权重形状: {weight.shape}")
    
    # 如果是3D张量，选择第一个专家
    if weight.ndim == 3:
        expert_idx = 0
        expert_weight = weight[expert_idx].float().numpy()
        print(f"选择专家 {expert_idx}, 形状: {expert_weight.shape}")
    else:
        expert_weight = weight.float().numpy()
    
    # 测试不同码本大小（为了快速验证，只测试256）
    codebook_sizes = [256]
    results = {}
    
    for codebook_size in codebook_sizes:
        print(f"\n{'='*50}")
        print(f"测试码本大小: {codebook_size}")
        print(f"{'='*50}")
        
        # 分析RVQ分层
        analysis = analyze_rvq_levels(
            expert_weight, 
            num_levels=4, 
            codebook_size=codebook_size
        )
        results[codebook_size] = analysis
        
        # 计算压缩比
        original_size = expert_weight.size
        # 每个向量需要log2(codebook_size)位来编码，共4级
        bits_per_code = np.log2(codebook_size)
        compressed_size = expert_weight.shape[0] * 4 * bits_per_code / 8  # 字节
        compression_ratio = original_size / compressed_size
        
        print(f"\n压缩比: {compression_ratio:.2f}x")
        print(f"原始大小: {original_size} 元素")
        print(f"压缩后大小: {compressed_size:.0f} 字节")
    
    # 测试PCA白化预处理
    print("\n" + "=" * 70)
    print("测试PCA白化预处理后的RVQ")
    print("=" * 70)
    
    # PCA白化
    whitened, pca, scaler = pca_whitening(expert_weight)
    
    # 分析白化后的RVQ
    analysis_whitened = analyze_rvq_levels(
        whitened, 
        num_levels=4, 
        codebook_size=256
    )
    
    # 比较白化前后的分层特性
    print("\n" + "=" * 70)
    print("白化前后分层特性比较")
    print("=" * 70)
    
    original_analysis = results[256]  # 码本大小256的结果
    
    print("\n原始权重:")
    for i, ratio in enumerate(original_analysis['energy_ratios']):
        print(f"  L{i+1}: {ratio*100:.2f}%")
        
    print("\n白化后权重:")
    for i, ratio in enumerate(analysis_whitened['energy_ratios']):
        print(f"  L{i+1}: {ratio*100:.2f}%")
    
    # 保存结果
    output_file = "rvq_semantic_layering_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Qwen3.5-35B-A3B Expert RVQ语义分层分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        for codebook_size, analysis in results.items():
            f.write(f"码本大小: {codebook_size}\n")
            f.write("-" * 30 + "\n")
            f.write("各级能量比例:\n")
            for i, ratio in enumerate(analysis['energy_ratios']):
                level_name = f"L{i+1}"
                if i == 0:
                    semantic = "任务相关"
                elif i == 1:
                    semantic = "句法结构"
                elif i == 2:
                    semantic = "知识细节"
                else:
                    semantic = "残差噪声"
                f.write(f"  {level_name} ({semantic}): {ratio*100:.2f}%\n")
            f.write("\n")
            
        f.write("\nPCA白化后分析:\n")
        f.write("-" * 30 + "\n")
        f.write("各级能量比例:\n")
        for i, ratio in enumerate(analysis_whitened['energy_ratios']):
            level_name = f"L{i+1}"
            if i == 0:
                semantic = "任务相关"
            elif i == 1:
                semantic = "句法结构"
            elif i == 2:
                semantic = "知识细节"
            else:
                semantic = "残差噪声"
            f.write(f"  {level_name} ({semantic}): {ratio*100:.2f}%\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 可视化
    visualize_rvq_analysis(results, analysis_whitened)
    
    return results, analysis_whitened

def visualize_rvq_analysis(results, analysis_whitened):
    """
    可视化RVQ分析结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 各级能量比例
    ax1 = axes[0, 0]
    codebook_sizes = list(results.keys())
    for i, codebook_size in enumerate(codebook_sizes):
        analysis = results[codebook_size]
        ratios = analysis['energy_ratios']
        ax1.bar([x + i*0.2 for x in range(len(ratios))], ratios, 
                width=0.2, label=f'码本{codebook_size}')
    
    ax1.set_xlabel('RVQ级别')
    ax1.set_ylabel('能量比例')
    ax1.set_title('各级能量比例')
    ax1.legend()
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(['L1', 'L2', 'L3', 'L4'])
    
    # 2. 重建误差
    ax2 = axes[0, 1]
    for i, codebook_size in enumerate(codebook_sizes):
        analysis = results[codebook_size]
        errors = [e*100 for e in analysis['errors']]
        ax2.plot(range(1, 5), errors, 'o-', label=f'码本{codebook_size}')
    
    ax2.set_xlabel('RVQ级别')
    ax2.set_ylabel('重建误差 (%)')
    ax2.set_title('各级重建误差')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 码本差异性
    ax3 = axes[1, 0]
    for i, codebook_size in enumerate(codebook_sizes):
        analysis = results[codebook_size]
        differences = analysis['codebook_differences']
        ax3.plot(range(1, len(differences)+1), differences, 'o-', 
                label=f'码本{codebook_size}')
    
    ax3.set_xlabel('级别')
    ax3.set_ylabel('平均距离')
    ax3.set_title('相邻级别码本差异')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 白化前后对比
    ax4 = axes[1, 1]
    original_ratios = results[256]['energy_ratios']
    whitened_ratios = analysis_whitened['energy_ratios']
    
    x = np.arange(4)
    width = 0.35
    ax4.bar(x - width/2, original_ratios, width, label='原始权重')
    ax4.bar(x + width/2, whitened_ratios, width, label='白化后权重')
    
    ax4.set_xlabel('RVQ级别')
    ax4.set_ylabel('能量比例')
    ax4.set_title('白化前后各级能量比例对比')
    ax4.legend()
    ax4.set_xticks(x)
    ax4.set_xticklabels(['L1', 'L2', 'L3', 'L4'])
    
    plt.tight_layout()
    plt.savefig('rvq_semantic_layering_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("可视化结果已保存到: rvq_semantic_layering_analysis.png")

if __name__ == "__main__":
    # 运行测试
    results, analysis_whitened = test_expert_rvq()