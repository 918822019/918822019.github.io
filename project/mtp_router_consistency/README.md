# MTP Router Consistency Analysis

对 BailingMoeV2 模型中 MTP（Multi-Token Prediction）模块的路由预测与 Decoder 实际路由之间的一致性进行多维度定量分析。

## 背景

BailingMoeV2 架构内建了 `BailingMoeV2MTPLayer`（即 MTP Head），其核心思想是：在 Decoder 预测下一个 token 的同时，MTP Head 也预测该 token 的 **MoE 专家路由分配**。通过对比 MTP 预测的 router logits 与 Decoder 实际产生的 router logits，可以评估 MTP 是否准确学到了 expert 选择模式。

## 架构概览

```
Input Tokens
    │
    ▼
Word Embedding
    │
    ▼
┌────────────────────────────────────┐
│ BailingMoeV2DecoderLayer × 20      │
│  ├─ Self-Attention                 │
│  ├─ MoE FFN                        │
│  │   └─ Router (top-8 of 256) ──── router_logits[0..18]
│  └─ Add & Norm                     │
└────────────────────────────────────┘
    │
    ▼
Final Norm
    │
    ├──────────────────────┐
    ▼                      ▼
LM Head              MTP Head (BailingMoeV2MTPLayer)
    │                      ├─ MTP Router ──── router_logits[-1]
    │                      ├─ Self-Attention
    ▼                      ▼
next token           next token's routing prediction
```

**关键关系**：
- `router_logits[-1]` = MTP Router 的输出（预测下一位置的 expert 路由）
- `router_logits[-2]` = 最后一层 Decoder Router 的输出（当前 token 的实际路由）
- 对比：`mtp_router[:, :-1, :]` vs `decoder_router[:, 1:, :]`
- 即 MTP 在位置 t 的输出，与 Decoder 在位置 t+1 的实际路由进行比较

## 项目结构

| 文件 | 作用 |
|---|---|
| `decoder.py` | 核心数据提取。自回归生成 + router logits 提取 + 对齐 |
| `compare.py` | 5 组对比指标函数 |
| `main.py` | 调度入口，加载模型 → 跑分析 → 生成报告 |
| `config.py` | 配置（模型路径、生成参数、MoE 参数） |
| `model_utils.py` | HuggingFace 模型/分词器加载 |
| `mtp_head.py` | 说明 MTP 内建于 BailingMoeV2 架构 |
| `check_mtp.py` | 检查模型权重中 MTP 相关层的 keys |
| `check_routing.py` | 独立工具：分析同一 token 在不同位置的路由一致性 |
| `download_model.py` | ModelScope 下载 Mellum2 模型 |
| `requirements.txt` | torch, transformers |

## 分析指标

### 0) Router 一致性（基础指标）

对每对 `(MTP_pred, actual_decoder)` router logits 计算：

| 指标 | 含义 |
|---|---|
| `kl_div` | KL 散度，分布差异 |
| `js_div` | JS 散度，对称分布距离 |
| `cosine_sim` | 余弦相似度，向量方向一致性 |
| `top_k_iou` | Top-K Expert 集合的 IoU |
| `spearman_rho` | Spearman 秩相关，排序一致性 |
| `top_{1,3,5}_hit_rate` | MTP top-k 中是否包含实际 top-1 expert |

### 1) Router 置信度 / 熵分析

| 指标 | 含义 |
|---|---|
| `mtp_entropy_mean` | MTP Router softmax 熵均值 |
| `actual_entropy_mean` | Decoder Router softmax 熵均值 |
| `entropy_diff_mean` | 熵差（MTP - Decoder） |
| `mtp_confidence_mean` | MTP Router top-1 概率均值 |
| `actual_confidence_mean` | Decoder Router top-1 概率均值 |
| `confidence_diff_mean` | 置信度差 |

### 2) Expert 命中分析

| 指标 | 含义 |
|---|---|
| `avg_overlap_count` | top-k 中重叠 expert 平均数 |
| `zero_overlap_ratio` | 完全不命中比例 |
| `full_overlap_ratio` | 完全命中比例 |

### 3) 逐层 Router 对比

将 MTP Router 与 **每一层** Decoder Layer 的 Router 分别计算 Cosine 和 IoU，观察 MTP 的路由预测与哪一层最接近。

- `layerwise_cosine`：每层的 Cosine，长度 = num_layers
- `layerwise_cosine_mean`：逐层均值
- `last_layer_cosine`：最后一层（即 `router_logits[-2]`）
- `layerwise_cosine_improvement`：末层 - 首层差值

### 4) LM Head vs MTP Router 置信度关联

| 指标 | 含义 |
|---|---|
| `lm_mtp_confidence_corr` | LM Head top-1 置信度与 MTP Router 置信度的 Pearson 相关系数 |
| `lm_confidence_mean` | LM Head 平均置信度 |

### 5) 位置趋势分析

将生成序列等分为前后两半，比较每个指标在前半和后半的均值，观察是否随生成位置变化。

## 用法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行分析
py main.py

# 输出
output/
├── report.md     # 多维度分析报告
└── results.json  # 完整分析数据
```

### 配置

编辑 `config.py` 中的 `Config` 类：

```python
@dataclass
class Config:
    model_path: str = "data/models/Ling-mini-base-2.0"  # 模型路径
    max_new_tokens: int = 4     # 生成步数（影响比较对数）
    max_prompt_len: int = 64    # 截断 prompt 长度
    num_experts: int = 256      # MoE 专家数
    num_experts_per_tok: int = 8  # top-k routing
    prompts: list = [...]       # 测试 prompt 列表
```

### 独立工具

```bash
# 检查 MTP 权重 keys
py check_mtp.py

# 检查同一 token 不同位置的路由一致性
py check_routing.py
```

## 分析结果示例

在 Ling-mini-base-2.0 上的测试结果（max_new_tokens=4）：

| 维度 | 关键发现 |
|---|---|
| Router 一致性 | Cosine ~0.86，但 top-8 IoU 仅 0.02，hit_rate=0 |
| 置信度/熵 | MTP 熵（~3.1）> Decoder（~2.7），MTP 更不确定 |
| Expert 命中 | 平均仅重叠 0.5/8 个 expert，50%~87% 位置完全不命中 |
| 逐层对比 | MTP 与浅层 Decoder 最相似（Cosine ~0.92），随层深降至 ~0.86 |
| LM 关联 | LM 置信度与 MTP 置信度呈弱负相关（~-0.2） |

结论：MTP 预测的路由分布与 Decoder 在**分布层面**有中等相似性（Cosine），但在 **expert 选择层面**几乎无法准确命中。
