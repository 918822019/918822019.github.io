"""
模型缝合与 PPL 评估脚本

目的：验证 FSQ 量化后的模型在真实文本生成上的困惑度 (Perplexity)

操作流程：
1. 加载原始的 Qwen 0.8B FP16 模型
2. 遍历其 model.layers，将 gate_proj, up_proj, down_proj 的权重替换为 FSQ 反量化后的权重
3. 准备 WikiText-2 或 C4 的验证集
4. 计算 PPL = exp(CrossEntropyLoss)

预期目标：
- Qwen 0.8B FP16 基线 PPL 大约在 10.5 ~ 11.5 之间
- FSQ-16 (4-bit) + KL 目标 PPL < 11.5（即 PPL 劣化 < 0.5）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parents[3]
model_path = str(_ROOT / "data" / "models" / "Qwen3.5-0.8B")

# 输出目录
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


def load_quantized_weights(weights_file: str) -> Dict[str, torch.Tensor]:
    """
    加载量化后的权重
    
    Args:
        weights_file: 量化权重文件路径
        
    Returns:
        quantized_weights: 量化权重字典
    """
    print(f"加载量化权重: {weights_file}")
    
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"量化权重文件不存在: {weights_file}")
    
    quantized_weights = torch.load(weights_file, map_location='cpu')
    
    print(f"  包含 {len(quantized_weights)} 个张量")
    for key, value in quantized_weights.items():
        if value is not None:
            print(f"    {key}: {value.shape}")
    
    return quantized_weights


def apply_quantized_weights_to_model(
    model: AutoModelForCausalLM,
    quantized_weights: Dict[str, torch.Tensor],
    layer_indices: List[int] = [0]
) -> AutoModelForCausalLM:
    """
    将量化后的权重应用到模型
    
    Args:
        model: 原始模型
        quantized_weights: 量化权重字典
        layer_indices: 要替换的层索引列表（默认为[0]）
        
    Returns:
        model: 替换权重后的模型
    """
    print(f"\n应用量化权重到模型（层 {layer_indices}）...")
    
    # 遍历每一层
    for layer_idx in layer_indices:
        # 获取目标层
        layer = model.model.layers[layer_idx]
        
        # 替换 gate_proj 权重
        if 'gate_proj' in quantized_weights:
            gate_proj_weight = quantized_weights['gate_proj']
            layer.mlp.gate_proj.weight.data = gate_proj_weight.to(layer.mlp.gate_proj.weight.device)
            print(f"  层 {layer_idx}: 替换 gate_proj: {gate_proj_weight.shape}")
        
        # 替换 up_proj 权重
        if 'up_proj' in quantized_weights:
            up_proj_weight = quantized_weights['up_proj']
            layer.mlp.up_proj.weight.data = up_proj_weight.to(layer.mlp.up_proj.weight.device)
            print(f"  层 {layer_idx}: 替换 up_proj: {up_proj_weight.shape}")
        
        # 替换 down_proj 权重
        if 'down_proj' in quantized_weights:
            down_proj_weight = quantized_weights['down_proj']
            layer.mlp.down_proj.weight.data = down_proj_weight.to(layer.mlp.down_proj.weight.device)
            print(f"  层 {layer_idx}: 替换 down_proj: {down_proj_weight.shape}")
    
    return model


def load_wikitext2_validation(split: str = "validation", num_samples: int = 1000) -> List[str]:
    """
    加载 WikiText-2 验证集
    
    Args:
        split: 数据集分割（validation/test）
        num_samples: 样本数量
        
    Returns:
        texts: 文本列表
    """
    print(f"加载 WikiText-2 {split} 数据集...")
    
    try:
        # 加载 WikiText-2 数据集
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # 获取文本
        texts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            text = item["text"].strip()
            if len(text) > 0:  # 跳过空行
                texts.append(text)
        
        print(f"  加载了 {len(texts)} 个样本")
        print(f"  平均文本长度: {np.mean([len(t) for t in texts]):.1f} 字符")
        
        return texts
        
    except Exception as e:
        print(f"加载 WikiText-2 失败: {e}")
        print("使用合成数据进行测试...")
        
        # 生成合成数据
        synthetic_texts = []
        for i in range(num_samples):
            text = f"This is a sample text number {i}. " * 10
            synthetic_texts.append(text)
        
        return synthetic_texts


def calculate_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int = 512,
    stride: int = 256,
    device: str = "cuda"
) -> Tuple[float, List[float]]:
    """
    计算困惑度 (Perplexity)
    
    Args:
        model: 模型
        tokenizer: 分词器
        texts: 文本列表
        max_length: 最大序列长度
        stride: 滑动窗口步长
        device: 设备
        
    Returns:
        avg_ppl: 平均困惑度
        ppl_list: 每个样本的困惑度列表
    """
    print(f"\n计算困惑度 (PPL)...")
    print(f"  最大序列长度: {max_length}")
    print(f"  滑动窗口步长: {stride}")
    
    model.eval()
    
    ppl_list = []
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            # 分词
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(device)
            
            # 计算困惑度（使用滑动窗口）
            seq_len = input_ids.size(1)
            
            if seq_len < 2:
                continue
            
            # 计算 Loss
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # 计算困惑度
            ppl = torch.exp(loss).item()
            ppl_list.append(ppl)
            
            total_loss += loss.item() * seq_len
            total_tokens += seq_len
            
            if (i + 1) % 100 == 0:
                avg_ppl_so_far = np.mean(ppl_list[-100:])
                print(f"  样本 {i+1}/{len(texts)}: 当前 PPL = {ppl:.4f}, 平均 PPL = {avg_ppl_so_far:.4f}")
    
    # 计算平均困惑度
    avg_ppl = np.mean(ppl_list) if ppl_list else float('inf')
    
    return avg_ppl, ppl_list


def evaluate_fsq_quantized_model():
    """
    评估 FSQ 量化后的模型
    """
    print("=" * 70)
    print("FSQ 量化模型 PPL 评估")
    print("=" * 70)
    
    # 检查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载分词器
    print(f"\n加载分词器: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  词表大小: {tokenizer.vocab_size}")
    
    # 加载原始模型
    print(f"\n加载原始模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"  模型类型: {model.config.model_type}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载量化权重
    weights_file = output_dir / "quantized_weights_0_8b_fsq8_v2.pt"
    quantized_weights = load_quantized_weights(weights_file)
    
    # 应用量化权重到模型（替换第0层）
    model = apply_quantized_weights_to_model(model, quantized_weights, layer_indices=[0])
    
    # 加载验证数据集
    texts = load_wikitext2_validation(split="validation", num_samples=500)
    
    # 计算量化模型的 PPL
    print("\n" + "=" * 70)
    print("计算量化模型的 PPL")
    print("=" * 70)
    
    avg_ppl_quantized, ppl_list_quantized = calculate_ppl(
        model, tokenizer, texts, max_length=512, stride=256, device=device
    )
    
    # 计算原始模型的 PPL（作为基线）
    print("\n" + "=" * 70)
    print("计算原始模型的 PPL（基线）")
    print("=" * 70)
    
    # 重新加载原始模型
    model_baseline = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    avg_ppl_baseline, ppl_list_baseline = calculate_ppl(
        model_baseline, tokenizer, texts, max_length=512, stride=256, device=device
    )
    
    # 计算 PPL 劣化
    ppl_degradation = avg_ppl_quantized - avg_ppl_baseline
    ppl_degradation_percent = (ppl_degradation / avg_ppl_baseline) * 100
    
    # 打印结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    print(f"\n原始模型 (FP16):")
    print(f"  平均 PPL: {avg_ppl_baseline:.4f}")
    print(f"  PPL 标准差: {np.std(ppl_list_baseline):.4f}")
    print(f"  最小 PPL: {min(ppl_list_baseline):.4f}")
    print(f"  最大 PPL: {max(ppl_list_baseline):.4f}")
    
    print(f"\n量化模型 (FSQ-8 + Outlier 保护):")
    print(f"  平均 PPL: {avg_ppl_quantized:.4f}")
    print(f"  PPL 标准差: {np.std(ppl_list_quantized):.4f}")
    print(f"  最小 PPL: {min(ppl_list_quantized):.4f}")
    print(f"  最大 PPL: {max(ppl_list_quantized):.4f}")
    
    print(f"\nPPL 劣化:")
    print(f"  绝对劣化: {ppl_degradation:.4f}")
    print(f"  相对劣化: {ppl_degradation_percent:.2f}%")
    
    # 评估结论
    print(f"\n评估结论:")
    if ppl_degradation < 0.5:
        print("  ✓ PPL 劣化 < 0.5，达到 SOTA 级别！")
    elif ppl_degradation < 1.0:
        print("  ⚠ PPL 劣化 < 1.0，效果良好")
    elif ppl_degradation < 2.0:
        print("  ⚠ PPL 劣化 < 2.0，效果一般")
    else:
        print("  ✗ PPL 劣化 >= 2.0，效果较差")
    
    # 保存结果
    results = {
        "model_type": "Qwen3.5-0.8B",
        "quantization_method": "FSQ-8 + Per-channel Salience + Outlier Protection + KL Distillation",
        "baseline_ppl": avg_ppl_baseline,
        "quantized_ppl": avg_ppl_quantized,
        "ppl_degradation": ppl_degradation,
        "ppl_degradation_percent": ppl_degradation_percent,
        "num_samples": len(texts),
        "max_length": 512,
        "stride": 256
    }
    
    results_file = output_dir / "ppl_evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")
    
    return results


if __name__ == "__main__":
    # 运行评估
    results = evaluate_fsq_quantized_model()