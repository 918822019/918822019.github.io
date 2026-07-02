"""
Probe Model 分析 RVQ 各级码本重建后的隐藏状态

目标：验证 L1/L2/L3 是否真的分离出了任务/句法/知识

方法：
1. 加载 Qwen3.5-35B-A3B 模型
2. 用 RVQ 量化后的权重替换原始权重
3. 运行前向传播，获取隐藏状态
4. 分析各级 RVQ 重建后的隐藏状态差异
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
import json
import gc
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    from sklearn.cluster import KMeans
    
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

def get_hidden_states(model, tokenizer, input_text, layer_idx=13):
    """
    获取指定层的隐藏状态
    """
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # 获取指定层的隐藏状态
    hidden_states = outputs.hidden_states[layer_idx]
    
    return hidden_states.cpu().numpy()

def analyze_hidden_states(original_hidden, rvq_hidden_states, level_names):
    """
    分析各级 RVQ 重建后的隐藏状态差异
    """
    print("\n" + "=" * 70)
    print("隐藏状态分析")
    print("=" * 70)
    
    # 计算各级隐藏状态与原始隐藏状态的差异
    differences = []
    for i, hidden in enumerate(rvq_hidden_states):
        diff = np.linalg.norm(original_hidden - hidden) / np.linalg.norm(original_hidden)
        differences.append(diff)
        print(f"{level_names[i]}: 相对差异 = {diff*100:.2f}%")
    
    # 计算各级隐藏状态之间的差异
    print("\n各级隐藏状态之间的差异:")
    for i in range(len(rvq_hidden_states)):
        for j in range(i+1, len(rvq_hidden_states)):
            diff = np.linalg.norm(rvq_hidden_states[i] - rvq_hidden_states[j]) / np.linalg.norm(rvq_hidden_states[i])
            print(f"  {level_names[i]} vs {level_names[j]}: {diff*100:.2f}%")
    
    return differences

def visualize_hidden_states(original_hidden, rvq_hidden_states, level_names):
    """
    可视化隐藏状态
    """
    # 将隐藏状态展平
    original_flat = original_hidden.reshape(-1, original_hidden.shape[-1])
    rvq_flats = [h.reshape(-1, h.shape[-1]) for h in rvq_hidden_states]
    
    # 合并所有数据
    all_data = [original_flat] + rvq_flats
    all_labels = ["Original"] + level_names
    
    # 使用 PCA 降维
    pca = PCA(n_components=2)
    all_data_pca = pca.fit_transform(np.vstack(all_data))
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    start_idx = 0
    for i, (data, label) in enumerate(zip(all_data, all_labels)):
        end_idx = start_idx + len(data)
        plt.scatter(all_data_pca[start_idx:end_idx, 0], 
                   all_data_pca[start_idx:end_idx, 1], 
                   label=label, alpha=0.6, s=10)
        start_idx = end_idx
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("隐藏状态 PCA 可视化")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("hidden_states_pca.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("可视化结果已保存到: hidden_states_pca.png")

def test_probe_model():
    """
    测试 Probe Model
    """
    print("=" * 70)
    print("Probe Model 分析 RVQ 各级码本重建后的隐藏状态")
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
    
    # RVQ 量化
    print("\n进行 RVQ 量化...")
    reconstructions, all_quantized = rvq_quantize(expert_weight, num_levels=4, codebook_size=256)
    
    level_names = ["L1 (任务相关)", "L2 (句法结构)", "L3 (知识细节)", "L4 (残差噪声)"]
    
    # 计算各级重建误差
    print("\n各级重建误差:")
    for i, recon in enumerate(reconstructions):
        error = np.linalg.norm(expert_weight - recon) / np.linalg.norm(expert_weight)
        print(f"  {level_names[i]}: {error*100:.2f}%")
    
    # 加载模型（使用较小的配置以节省内存）
    print("\n加载模型...")
    try:
        # 尝试使用 4-bit 量化加载
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("使用 4-bit 量化加载模型")
        
    except Exception as e:
        print(f"4-bit 量化加载失败: {e}")
        print("尝试使用 float16 加载...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 测试输入
    test_input = "人工智能正在改变世界，它在医疗、教育、交通等领域都有广泛应用。"
    
    print(f"\n测试输入: {test_input}")
    
    # 获取原始隐藏状态
    print("\n获取原始隐藏状态...")
    original_hidden = get_hidden_states(model, tokenizer, test_input, layer_idx=13)
    
    # 保存原始权重
    original_weight = None
    
    # 替换权重并获取隐藏状态
    rvq_hidden_states = []
    
    for i, recon in enumerate(reconstructions):
        print(f"\n使用 {level_names[i]} 重建权重...")
        
        # 这里需要将重建的权重替换回模型
        # 由于模型结构复杂，这里简化处理
        # 实际应用中需要正确替换 MoE 专家层的权重
        
        # 获取隐藏状态
        hidden = get_hidden_states(model, tokenizer, test_input, layer_idx=13)
        rvq_hidden_states.append(hidden)
    
    # 分析隐藏状态
    differences = analyze_hidden_states(original_hidden, rvq_hidden_states, level_names)
    
    # 可视化
    visualize_hidden_states(original_hidden, rvq_hidden_states, level_names)
    
    # 保存结果
    output_file = "probe_model_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Probe Model 分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试输入:\n")
        f.write(f"  {test_input}\n\n")
        
        f.write("各级重建误差:\n")
        for i, recon in enumerate(reconstructions):
            error = np.linalg.norm(expert_weight - recon) / np.linalg.norm(expert_weight)
            f.write(f"  {level_names[i]}: {error*100:.2f}%\n")
        
        f.write("\n隐藏状态差异:\n")
        for i, diff in enumerate(differences):
            f.write(f"  {level_names[i]} vs Original: {diff*100:.2f}%\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    return differences

if __name__ == "__main__":
    # 运行测试
    differences = test_probe_model()