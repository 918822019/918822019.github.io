"""
验证指标实现

包括：
1. Router 一致性验证
2. Per-Token PPL 分布
3. L1 码字语义探针
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import json
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"


class RouterConsistencyMetric:
    """
    Router 一致性验证
    
    检查量化后 Expert 被选中的 Top-K 集合与 FP16 的重合率
    若 <95%，说明 FSQ 破坏了路由生态
    """
    
    def __init__(self, top_k: int = 2):
        self.top_k = top_k
    
    def compute(
        self,
        router_logits_original: torch.Tensor,
        router_logits_quantized: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算 Router 一致性
        
        Args:
            router_logits_original: 原始模型的 Router logits
            router_logits_quantized: 量化模型的 Router logits
            
        Returns:
            metrics: 指标字典
        """
        # 获取 Top-K 索引
        _, top_k_original = torch.topk(router_logits_original, self.top_k, dim=-1)
        _, top_k_quantized = torch.topk(router_logits_quantized, self.top_k, dim=-1)
        
        # 计算重合率
        # 对于每个样本，检查 Top-K 集合是否相同
        matches = 0
        total = top_k_original.shape[0]
        
        for i in range(total):
            set_original = set(top_k_original[i].cpu().numpy())
            set_quantized = set(top_k_quantized[i].cpu().numpy())
            
            if set_original == set_quantized:
                matches += 1
        
        consistency = matches / total
        
        # 计算平均重合数
        overlap_counts = []
        for i in range(total):
            set_original = set(top_k_original[i].cpu().numpy())
            set_quantized = set(top_k_quantized[i].cpu().numpy())
            overlap = len(set_original & set_quantized)
            overlap_counts.append(overlap)
        
        avg_overlap = np.mean(overlap_counts)
        
        return {
            "router_consistency": consistency,
            "avg_overlap": avg_overlap,
            "top_k": self.top_k
        }


class RoutingWeightJSDivergence:
    """
    Routing Weight JS 散度验证
    
    增强指标：不仅检查 Top-K 集合是否一致，还检查权重分数是否偏移
    例如：原本 Expert A 占 60%、Expert B 占 40%，量化后变成各 50%
    Top-K 没变，但输出已严重偏离
    
    目标：D_JS(w_fp16 || w_fsq) < 0.01
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def compute(
        self,
        router_logits_original: torch.Tensor,
        router_logits_quantized: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算 Routing Weight JS 散度
        
        Args:
            router_logits_original: 原始模型的 Router logits
            router_logits_quantized: 量化模型的 Router logits
            
        Returns:
            metrics: 指标字典
        """
        # 计算 Routing Weights (softmax)
        weights_original = F.softmax(router_logits_original, dim=-1)
        weights_quantized = F.softmax(router_logits_quantized, dim=-1)
        
        # 计算 JS 散度
        # JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        # 其中 M = 0.5 * (P + Q)
        
        # 计算平均分布 M
        M = 0.5 * (weights_original + weights_quantized)
        
        # 计算 KL(P || M)
        kl_original = torch.sum(
            weights_original * torch.log((weights_original + self.epsilon) / (M + self.epsilon)),
            dim=-1
        )
        
        # 计算 KL(Q || M)
        kl_quantized = torch.sum(
            weights_quantized * torch.log((weights_quantized + self.epsilon) / (M + self.epsilon)),
            dim=-1
        )
        
        # 计算 JS 散度
        js_divergence = 0.5 * (kl_original + kl_quantized)
        
        # 计算统计量
        js_np = js_divergence.cpu().numpy()
        
        metrics = {
            "avg_js_divergence": float(np.mean(js_np)),
            "std_js_divergence": float(np.std(js_np)),
            "max_js_divergence": float(np.max(js_np)),
            "min_js_divergence": float(np.min(js_np)),
            "js_divergence_95th": float(np.percentile(js_np, 95)),
            "js_divergence_99th": float(np.percentile(js_np, 99)),
            "num_samples": len(js_np)
        }
        
        return metrics
    
    def check_threshold(
        self,
        router_logits_original: torch.Tensor,
        router_logits_quantized: torch.Tensor,
        threshold: float = 0.01
    ) -> Tuple[bool, Dict[str, float]]:
        """
        检查 JS 散度是否在阈值内
        
        Args:
            router_logits_original: 原始模型的 Router logits
            router_logits_quantized: 量化模型的 Router logits
            threshold: 阈值
            
        Returns:
            passed: 是否通过
            metrics: 指标字典
        """
        metrics = self.compute(router_logits_original, router_logits_quantized)
        passed = metrics["avg_js_divergence"] < threshold
        
        return passed, metrics


class PerTokenPPLMetric:
    """
    Per-Token PPL 分布
    
    不要只看平均 PPL。画出 PPL 的长尾分布。
    FSQ 可能在常见 token 上表现完美，但在罕见实体/代码符号上灾难性退化。
    """
    
    def __init__(self):
        pass
    
    def compute(
        self,
        logits_original: torch.Tensor,
        logits_quantized: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算 Per-Token PPL
        
        Args:
            logits_original: 原始模型的 Logits
            logits_quantized: 量化模型的 Logits
            targets: 目标 token
            
        Returns:
            metrics: 指标字典
        """
        # 计算交叉熵损失
        ce_original = F.cross_entropy(
            logits_original.view(-1, logits_original.shape[-1]),
            targets.view(-1),
            reduction='none'
        )
        
        ce_quantized = F.cross_entropy(
            logits_quantized.view(-1, logits_quantized.shape[-1]),
            targets.view(-1),
            reduction='none'
        )
        
        # 计算 PPL
        ppl_original = torch.exp(ce_original)
        ppl_quantized = torch.exp(ce_quantized)
        
        # 计算统计量
        ppl_original_np = ppl_original.cpu().numpy()
        ppl_quantized_np = ppl_quantized.cpu().numpy()
        
        # 计算 PPL 差异
        ppl_diff = ppl_quantized_np - ppl_original_np
        
        metrics = {
            "avg_ppl_original": float(np.mean(ppl_original_np)),
            "avg_ppl_quantized": float(np.mean(ppl_quantized_np)),
            "avg_ppl_diff": float(np.mean(ppl_diff)),
            "std_ppl_diff": float(np.std(ppl_diff)),
            "max_ppl_diff": float(np.max(ppl_diff)),
            "min_ppl_diff": float(np.min(ppl_diff)),
            "ppl_diff_95th": float(np.percentile(ppl_diff, 95)),
            "ppl_diff_99th": float(np.percentile(ppl_diff, 99))
        }
        
        return metrics


class L1CodebookSemanticProbe:
    """
    L1 码字语义探针
    
    随机采样 100 个 L1 码字，反查它们最常出现在哪些 token/任务的激活中。
    如果 L1 码字能聚类出"数学符号"、"中文动词"、"代码关键字"等语义簇，那就彻底证实了语义分层。
    """
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
    
    def analyze(
        self,
        codebook: torch.Tensor,
        activations: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        分析 L1 码字的语义
        
        Args:
            codebook: L1 码本
            activations: 激活值
            token_ids: token IDs（可选）
            
        Returns:
            analysis: 分析结果
        """
        print(f"分析 L1 码字语义（{self.num_samples} 个样本）...")
        
        # 随机采样码字
        if len(codebook) > self.num_samples:
            indices = torch.randperm(len(codebook))[:self.num_samples]
            sampled_codebook = codebook[indices]
        else:
            sampled_codebook = codebook
            indices = torch.arange(len(codebook))
        
        # 计算每个码字与激活值的相似度
        # 使用余弦相似度
        similarities = F.cosine_similarity(
            sampled_codebook.unsqueeze(1),  # (num_samples, 1, dim)
            activations.unsqueeze(0),  # (1, num_ samples, dim)
            dim=-1
        )  # (num_samples, num_samples)
        
        # 找到每个码字最相似的激活值
        max_similarities, max_indices = similarities.max(dim=1)
        
        # 分析码字的分布
        codebook_norms = torch.norm(sampled_codebook, dim=-1)
        
        # 如果有 token IDs，分析码字与 token 的关系
        token_analysis = None
        if token_ids is not None:
            token_analysis = self._analyze_token_associations(
                sampled_codebook, activations, token_ids, max_indices
            )
        
        analysis = {
            "num_codebook_words": len(sampled_codebook),
            "avg_codebook_norm": float(codebook_norms.mean()),
            "std_codebook_norm": float(codebook_norms.std()),
            "avg_max_similarity": float(max_similarities.mean()),
            "std_max_similarity": float(max_similarities.std()),
            "token_analysis": token_analysis
        }
        
        return analysis
    
    def _analyze_token_associations(
        self,
        codebook: torch.Tensor,
        activations: torch.Tensor,
        token_ids: torch.Tensor,
        max_indices: torch.Tensor
    ) -> Dict[str, any]:
        """
        分析码字与 token 的关联
        """
        # 统计每个码字最常关联的 token
        token_counts = defaultdict(lambda: defaultdict(int))
        
        for i, idx in enumerate(max_indices):
            token_id = token_ids[idx].item()
            token_counts[i][token_id] += 1
        
        # 分析码字的语义簇
        # 这里简化处理，实际应用中可以使用聚类算法
        clusters = {
            "math_symbols": [],
            "chinese_verbs": [],
            "code_keywords": [],
            "other": []
        }
        
        # 假设的 token ID 范围（需要根据实际词表调整）
        math_range = range(0, 1000)
        chinese_range = range(1000, 5000)
        code_range = range(5000, 10000)
        
        for i, token_dict in token_counts.items():
            if not token_dict:
                continue
            
            # 找到最常见的 token
            most_common_token = max(token_dict.items(), key=lambda x: x[1])[0]
            
            if most_common_token in math_range:
                clusters["math_symbols"].append(i)
            elif most_common_token in chinese_range:
                clusters["chinese_verbs"].append(i)
            elif most_common_token in code_range:
                clusters["code_keywords"].append(i)
            else:
                clusters["other"].append(i)
        
        return {
            "clusters": clusters,
            "num_math_symbols": len(clusters["math_symbols"]),
            "num_chinese_verbs": len(clusters["chinese_verbs"]),
            "num_code_keywords": len(clusters["code_keywords"]),
            "num_other": len(clusters["other"])
        }


def load_expert_weights() -> Dict[str, torch.Tensor]:
    """
    加载 Expert 权重
    """
    print("加载 Expert 权重...")
    
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    
    moe_keys = [k for k in weight_map.keys() if "experts" in k]
    if not moe_keys:
        moe_keys = [k for k in weight_map.keys() if "gate" in k or "up" in k or "down" in k]
    
    gate_up_key = None
    down_key = None
    
    for key in moe_keys:
        if "gate_up_proj" in key:
            gate_up_key = key
        elif "down_proj" in key:
            down_key = key
    
    if gate_up_key is None or down_key is None:
        raise ValueError("未找到 gate_up_proj 或 down_proj")
    
    weights = {}
    
    for key in [gate_up_key, down_key]:
        file_name = weight_map[key]
        file_path = f"{model_path}/{file_name}"
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            weights[key] = f.get_tensor(key)
            print(f"  加载 {key}: {weights[key].shape}")
    
    return weights


def test_verification_metrics():
    """
    测试验证指标
    """
    print("=" * 70)
    print("验证指标测试")
    print("=" * 70)
    
    # 加载权重
    weights = load_expert_weights()
    
    gate_up_key = [k for k in weights.keys() if "gate_up_proj" in k][0]
    down_key = [k for k in weights.keys() if "down_proj" in k][0]
    
    # 模拟数据
    batch_size = 32
    seq_len = 128
    hidden_size = weights[gate_up_key].shape[2]
    num_experts = weights[gate_up_key].shape[0]
    vocab_size = 32000
    
    # 创建模拟数据
    router_logits_original = torch.randn(batch_size, num_experts)
    router_logits_quantized = router_logits_original + torch.randn_like(router_logits_original) * 0.1
    
    logits_original = torch.randn(batch_size, seq_len, vocab_size)
    logits_quantized = logits_original + torch.randn_like(logits_original) * 0.1
    
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 测试 Router 一致性
    print("\n" + "=" * 50)
    print("Router 一致性测试")
    print("=" * 50)
    
    router_metric = RouterConsistencyMetric(top_k=2)
    router_metrics = router_metric.compute(router_logits_original, router_logits_quantized)
    
    print(f"Router 一致性: {router_metrics['router_consistency']*100:.2f}%")
    print(f"平均重合数: {router_metrics['avg_overlap']:.2f}")
    print(f"Top-K: {router_metrics['top_k']}")
    
    # 测试 Routing Weight JS 散度
    print("\n" + "=" * 50)
    print("Routing Weight JS 散度测试")
    print("=" * 50)
    
    js_metric = RoutingWeightJSDivergence()
    js_metrics = js_metric.compute(router_logits_original, router_logits_quantized)
    
    print(f"平均 JS 散度: {js_metrics['avg_js_divergence']:.6f}")
    print(f"JS 散度标准差: {js_metrics['std_js_divergence']:.6f}")
    print(f"最大 JS 散度: {js_metrics['max_js_divergence']:.6f}")
    print(f"95th JS 散度: {js_metrics['js_divergence_95th']:.6f}")
    
    # 检查阈值
    passed, _ = js_metric.check_threshold(router_logits_original, router_logits_quantized, threshold=0.01)
    print(f"是否通过阈值检查 (JS < 0.01): {'是' if passed else '否'}")
    
    # 测试 Per-Token PPL
    print("\n" + "=" * 50)
    print("Per-Token PPL 测试")
    print("=" * 50)
    
    ppl_metric = PerTokenPPLMetric()
    ppl_metrics = ppl_metric.compute(logits_original, logits_quantized, targets)
    
    print(f"平均 PPL (原始): {ppl_metrics['avg_ppl_original']:.2f}")
    print(f"平均 PPL (量化): {ppl_metrics['avg_ppl_quantized']:.2f}")
    print(f"平均 PPL 差异: {ppl_metrics['avg_ppl_diff']:.2f}")
    print(f"PPL 差异标准差: {ppl_metrics['std_ppl_diff']:.2f}")
    print(f"PPL 差异 95th: {ppl_metrics['ppl_diff_95th']:.2f}")
    print(f"PPL 差异 99th: {ppl_metrics['ppl_diff_99th']:.2f}")
    
    # 测试 L1 码字语义探针
    print("\n" + "=" * 50)
    print("L1 码字语义探针测试")
    print("=" * 50)
    
    # 创建模拟码本和激活值
    codebook_size = 16
    codebook = torch.randn(codebook_size, hidden_size)
    activations = torch.randn(1000, hidden_size)
    token_ids = torch.randint(0, vocab_size, (1000,))
    
    probe = L1CodebookSemanticProbe(num_samples=50)
    probe_metrics = probe.analyze(codebook, activations, token_ids)
    
    print(f"码字数量: {probe_metrics['num_codebook_words']}")
    print(f"平均码字范数: {probe_metrics['avg_codebook_norm']:.4f}")
    print(f"平均最大相似度: {probe_metrics['avg_max_similarity']:.4f}")
    
    if probe_metrics['token_analysis']:
        token_analysis = probe_metrics['token_analysis']
        print(f"\n语义簇分析:")
        print(f"  数学符号: {token_analysis['num_math_symbols']}")
        print(f"  中文动词: {token_analysis['num_chinese_verbs']}")
        print(f"  代码关键字: {token_analysis['num_code_keywords']}")
        print(f"  其他: {token_analysis['num_other']}")
    
    # 保存结果
    output_file = "verification_metrics_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("验证指标测试结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Router 一致性:\n")
        f.write(f"  一致性: {router_metrics['router_consistency']*100:.2f}%\n")
        f.write(f"  平均重合数: {router_metrics['avg_overlap']:.2f}\n")
        f.write(f"  Top-K: {router_metrics['top_k']}\n\n")
        
        f.write("Routing Weight JS 散度:\n")
        f.write(f"  平均 JS 散度: {js_metrics['avg_js_divergence']:.6f}\n")
        f.write(f"  JS 散度标准差: {js_metrics['std_js_divergence']:.6f}\n")
        f.write(f"  最大 JS 散度: {js_metrics['max_js_divergence']:.6f}\n")
        f.write(f"  95th JS 散度: {js_metrics['js_divergence_95th']:.6f}\n")
        f.write(f"  是否通过阈值检查 (JS < 0.01): {'是' if passed else '否'}\n\n")
        
        f.write("Per-Token PPL:\n")
        f.write(f"  平均 PPL (原始): {ppl_metrics['avg_ppl_original']:.2f}\n")
        f.write(f"  平均 PPL (量化): {ppl_metrics['avg_ppl_quantized']:.2f}\n")
        f.write(f"  平均 PPL 差异: {ppl_metrics['avg_ppl_diff']:.2f}\n")
        f.write(f"  PPL 差异标准差: {ppl_metrics['std_ppl_diff']:.2f}\n")
        f.write(f"  PPL 差异 95th: {ppl_metrics['ppl_diff_95th']:.2f}\n")
        f.write(f"  PPL 差异 99th: {ppl_metrics['ppl_diff_99th']:.2f}\n\n")
        
        f.write("L1 码字语义探针:\n")
        f.write(f"  码字数量: {probe_metrics['num_codebook_words']}\n")
        f.write(f"  平均码字范数: {probe_metrics['avg_codebook_norm']:.4f}\n")
        f.write(f"  平均最大相似度: {probe_metrics['avg_max_similarity']:.4f}\n")
        
        if probe_metrics['token_analysis']:
            token_analysis = probe_metrics['token_analysis']
            f.write(f"\n  语义簇分析:\n")
            f.write(f"    数学符号: {token_analysis['num_math_symbols']}\n")
            f.write(f"    中文动词: {token_analysis['num_chinese_verbs']}\n")
            f.write(f"    代码关键字: {token_analysis['num_code_keywords']}\n")
            f.write(f"    其他: {token_analysis['num_other']}\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    return {
        "router_metrics": router_metrics,
        "js_metrics": js_metrics,
        "ppl_metrics": ppl_metrics,
        "probe_metrics": probe_metrics
    }


if __name__ == "__main__":
    # 运行测试
    results = test_verification_metrics()