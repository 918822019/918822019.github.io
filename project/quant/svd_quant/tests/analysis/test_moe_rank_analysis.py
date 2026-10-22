import torch
import numpy as np
from safetensors import safe_open
from collections import defaultdict
import json
import gc

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"

def load_index():
    """加载模型索引"""
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        return json.load(f)

def get_layer_info(index_data):
    """获取层信息"""
    weight_map = index_data["weight_map"]
    
    # 分类层 - 使用实际的权重名称格式
    layers = {
        "attention_q": [],
        "attention_k": [],
        "attention_v": [],
        "attention_o": [],
        "moe_gate_up_proj": [],  # MoE专家的合并gate_up投影
        "moe_down_proj": [],     # MoE专家的down投影
        "shared_gate_proj": [],  # 共享专家
        "shared_up_proj": [],
        "shared_down_proj": [],
        "router": [],            # 路由器
    }
    
    for name in weight_map.keys():
        if "self_attn.q_proj" in name:
            layers["attention_q"].append(name)
        elif "self_attn.k_proj" in name:
            layers["attention_k"].append(name)
        elif "self_attn.v_proj" in name:
            layers["attention_v"].append(name)
        elif "self_attn.o_proj" in name:
            layers["attention_o"].append(name)
        elif "mlp.experts.gate_up_proj" in name:
            layers["moe_gate_up_proj"].append(name)
        elif "mlp.experts.down_proj" in name:
            layers["moe_down_proj"].append(name)
        elif "mlp.shared_expert.gate_proj" in name:
            layers["shared_gate_proj"].append(name)
        elif "mlp.shared_expert.up_proj" in name:
            layers["shared_up_proj"].append(name)
        elif "mlp.shared_expert.down_proj" in name:
            layers["shared_down_proj"].append(name)
        elif "mlp.gate" in name and "expert" not in name:
            layers["router"].append(name)
    
    return layers

def compute_rank(weight_np, energy_threshold=0.95):
    """计算矩阵的有效秩（基于能量保留）"""
    try:
        # SVD分解
        U, S, Vh = np.linalg.svd(weight_np, full_matrices=False)
        
        # 计算能量分布
        total_energy = np.sum(S**2)
        cumulative_energy = np.cumsum(S**2) / total_energy
        
        # 找到达到阈值的秩
        rank = np.searchsorted(cumulative_energy, energy_threshold) + 1
        
        return rank, len(S), S
    except Exception as e:
        print(f"SVD计算错误: {e}")
        return None, None, None

def load_and_analyze_weights(file_path, target_names, max_samples=5):
    """加载并分析权重"""
    results = []
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            # 筛选目标权重
            targets = [k for k in keys if any(t in k for t in target_names)]
            
            print(f"  文件 {file_path.split('/')[-1]} 中找到 {len(targets)} 个目标权重")
            
            # 采样
            if len(targets) > max_samples:
                targets = np.random.choice(targets, max_samples, replace=False).tolist()
            
            for name in targets:
                try:
                    weight = f.get_tensor(name)
                    weight_np = weight.float().numpy()
                    
                    # 处理3D张量（MoE专家层可能是3D）
                    if weight_np.ndim == 3:
                        # 对于3D张量，计算每个专家的平均秩
                        ranks = []
                        all_singular = []
                        for i in range(weight_np.shape[0]):
                            expert_weight = weight_np[i]
                            rank, total, sv = compute_rank(expert_weight)
                            if rank is not None:
                                ranks.append(rank)
                                all_singular.extend(sv[:10])  # 只保留前10个奇异值
                        
                        if ranks:
                            avg_rank = np.mean(ranks)
                            avg_total = total  # 所有专家形状相同
                            results.append({
                                "name": name,
                                "shape": weight_np.shape,
                                "rank": avg_rank,
                                "total": avg_total,
                                "rank_ratio": avg_rank / avg_total,
                                "singular_values": np.array(all_singular[:10])
                            })
                            print(f"  {name}: shape={weight_np.shape}, avg_rank={avg_rank:.1f}/{avg_total} ({avg_rank/avg_total*100:.1f}%)")
                    elif weight_np.ndim == 2:
                        # 计算秩
                        rank, total, singular_values = compute_rank(weight_np)
                        
                        if rank is not None:
                            results.append({
                                "name": name,
                                "shape": weight_np.shape,
                                "rank": rank,
                                "total": total,
                                "rank_ratio": rank / total,
                                "singular_values": singular_values
                            })
                            print(f"  {name}: shape={weight_np.shape}, rank={rank}/{total} ({rank/total*100:.1f}%)")
                    else:
                        print(f"  跳过{weight_np.ndim}D张量: {name} shape={weight_np.shape}")
                    
                    # 释放内存
                    del weight, weight_np
                    gc.collect()
                    
                except Exception as e:
                    print(f"  处理 {name} 时出错: {e}")
                    continue
                    
    except Exception as e:
        print(f"  加载文件 {file_path} 时出错: {e}")
    
    return results

def main():
    print("=" * 70)
    print("Qwen3.5-35B-A3B MoE模型秩分析")
    print("=" * 70)
    
    # 加载索引
    index_data = load_index()
    weight_map = index_data["weight_map"]
    
    # 获取层信息
    layers = get_layer_info(index_data)
    
    print("\n📊 模型结构概览:")
    print(f"总权重数: {len(weight_map)}")
    for layer_type, names in layers.items():
        if names:
            print(f"  {layer_type}: {len(names)} 个")
    
    # 分析关键层类型
    analysis_targets = [
        ("attention_k", "注意力K层"),
        ("attention_v", "注意力V层"),
        ("moe_gate_up_proj", "MoE专家Gate_Up层"),
        ("moe_down_proj", "MoE专家Down层"),
        ("shared_gate_proj", "共享专家Gate层"),
        ("shared_up_proj", "共享专家Up层"),
        ("shared_down_proj", "共享专家Down层"),
    ]
    
    all_results = defaultdict(list)
    
    print("\n🔍 开始分析各层秩特性...")
    print("-" * 70)
    
    for layer_type, description in analysis_targets:
        if not layers[layer_type]:
            print(f"\n⚠️  {description}: 未找到相关层")
            continue
            
        print(f"\n📈 分析 {description}:")
        
        # 获取包含这些权重的文件
        files_to_load = set()
        for name in layers[layer_type][:10]:  # 只取前10个采样
            if name in weight_map:
                files_to_load.add(weight_map[name])
        
        # 分析每个文件
        for file_name in files_to_load:
            file_path = f"{model_path}/{file_name}"
            results = load_and_analyze_weights(file_path, [layer_type], max_samples=3)
            all_results[layer_type].extend(results)
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("📊 秩分析汇总")
    print("=" * 70)
    
    summary = {}
    for layer_type, results in all_results.items():
        if results:
            ranks = [r["rank"] for r in results]
            ratios = [r["rank_ratio"] for r in results]
            
            summary[layer_type] = {
                "count": len(results),
                "avg_rank": np.mean(ranks),
                "min_rank": np.min(ranks),
                "max_rank": np.max(ranks),
                "avg_ratio": np.mean(ratios),
                "sample_shape": results[0]["shape"]
            }
    
    # 打印汇总表
    print(f"\n{'层类型':<25} {'数量':<8} {'平均秩':<12} {'平均比例':<12} {'形状'}")
    print("-" * 80)
    
    for layer_type in ["attention_k", "attention_v", "moe_gate_up_proj", 
                        "moe_down_proj", "shared_gate_proj", "shared_up_proj", "shared_down_proj"]:
        if layer_type in summary:
            s = summary[layer_type]
            print(f"{layer_type:<25} {s['count']:<8} {s['avg_rank']:<12.1f} {s['avg_ratio']*100:<12.1f}% {s['sample_shape']}")
    
    # 特别关注MoE专家层
    print("\n" + "=" * 70)
    print("🔬 MoE专家层 vs 共享专家层 对比")
    print("=" * 70)
    
    # MoE专家层
    if "moe_gate_up_proj" in summary:
        moe_gate_up_ratio = summary["moe_gate_up_proj"]["avg_ratio"]
        print(f"\nGate_Up投影:")
        print(f"  MoE专家层平均秩比例: {moe_gate_up_ratio*100:.1f}%")
        
        # 对应共享专家
        if "shared_gate_proj" in summary and "shared_up_proj" in summary:
            shared_gate_ratio = summary["shared_gate_proj"]["avg_ratio"]
            shared_up_ratio = summary["shared_up_proj"]["avg_ratio"]
            shared_avg = (shared_gate_ratio + shared_up_ratio) / 2
            diff = moe_gate_up_ratio - shared_avg
            print(f"  共享专家层平均秩比例: {shared_avg*100:.1f}%")
            print(f"  差异: {diff*100:+.1f}%")
    
    # Down投影
    if "moe_down_proj" in summary:
        moe_down_ratio = summary["moe_down_proj"]["avg_ratio"]
        print(f"\nDown投影:")
        print(f"  MoE专家层平均秩比例: {moe_down_ratio*100:.1f}%")
        
        if "shared_down_proj" in summary:
            shared_down_ratio = summary["shared_down_proj"]["avg_ratio"]
            diff = moe_down_ratio - shared_down_ratio
            print(f"  共享专家层平均秩比例: {shared_down_ratio*100:.1f}%")
            print(f"  差异: {diff*100:+.1f}%")
    
    # 保存详细结果
    output_file = "moe_rank_analysis_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Qwen3.5-35B-A3B MoE模型秩分析详细结果\n")
        f.write("=" * 70 + "\n\n")
        
        for layer_type, results in all_results.items():
            f.write(f"\n{layer_type}:\n")
            f.write("-" * 50 + "\n")
            for r in results:
                f.write(f"  {r['name']}:\n")
                f.write(f"    Shape: {r['shape']}\n")
                f.write(f"    Rank: {r['rank']}/{r['total']} ({r['rank_ratio']*100:.1f}%)\n")
                f.write(f"    Top 10 Singular Values: {r['singular_values'][:10]}\n")
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    # 关键发现
    print("\n" + "=" * 70)
    print("💡 关键发现")
    print("=" * 70)
    
    moe_types = ["moe_gate_up_proj", "moe_down_proj"]
    if any(t in summary for t in moe_types):
        moe_avg = np.mean([summary[t]["avg_ratio"] for t in moe_types if t in summary])
        print(f"\n1. MoE专家层平均秩比例: {moe_avg*100:.1f}%")
        
        if moe_avg < 0.5:
            print("   ✅ MoE专家层呈现低秩特性，适合SVD压缩！")
        elif moe_avg < 0.7:
            print("   ⚠️  MoE专家层中等秩，SVD压缩有一定效果")
        else:
            print("   ❌ MoE专家层接近满秩，SVD压缩效果有限")
    
    if "attention_k" in summary:
        attn_avg = np.mean([summary[t]["avg_ratio"] for t in ["attention_k", "attention_v"] if t in summary])
        print(f"\n2. 注意力层平均秩比例: {attn_avg*100:.1f}%")

if __name__ == "__main__":
    main()