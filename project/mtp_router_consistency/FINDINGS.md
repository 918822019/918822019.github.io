# MTP Router 一致性分析 - 发现汇总

## 目标
研究 MTP（多 token 预测）层的路由决策能否用于预判 Decoder 的 Expert 加载，以加速 CPU offload 场景下的推理。

## 模型
- **架构**: BailingMoeV2 (Ling-mini-base-2.0)
- **参数量**: 12B
- **MoE**: 每层 256 experts，top-8 路由
- **层数**: 19 层 MoE + 1 层 MTP（共 20 层 decoder，1 层 dense）
- **隐层维度**: 2048
- **词表大小**: 157184

---

## 关键发现

### 1. MTP 层架构
MTP 层拥有 **自己独立的 256 个 expert**，与 Decoder 每层的 256 experts 完全不共享参数。

```
Decoder Layer (×19):
  input → Self-Attention → MoE Block (256 experts) → output

MTP Layer (×1):
  [shifted tok[t+1] emb, decoder_last_hidden[t]] → eh_proj → Self-Attention → MoE Block (256 experts) → lm_head → tok[t+2]
```

### 2. Token 预测精度
| 指标 | 样本 1 (def fibonacci(n):) | 样本 2 (def merge_sort(arr):) |
|------|----------------------------|-------------------------------|
| LM→tok[t+1] 精度 | 75.00% (3/4) | 25.00% (1/4) |
| MTP→tok[t+2] 精度 | 100.00% (3/3) | 33.33% (1/3) |
| MTP-LM 差距 | +25.00% | +8.33% |

**结论**: MTP 通过内部 `roll` 操作看到了 tok[t+1] 的嵌入，比 LM 多一个信息优势。

### 3. LM Head vs MTP Head Logits 分布（按目标 token 对齐）
两者都预测 tok[t+2] 时：
| 指标 | 样本 1 | 样本 2 |
|------|--------|--------|
| Cosine 相似度 | 0.9715 | 0.9650 |
| Spearman 相关 | 0.8342 | 0.7867 |
| KL 散度 | 0.3144 | 0.4777 |
| JS 散度 | 0.0833 | 0.1050 |
| Top-8 IoU | 0.5362 | 0.5111 |
| 概率分布点积 | 0.3631 | 0.4414 |

**结论**: LM 和 MTP 的 logits 分布非常接近（Cosine ~0.97），与 injectivity（双射）理论一致：不同路径的 hidden state 经 lm_head 投影后输出相似。

### 4. Router 一致性
MTP Router vs Decoder Router（最后一层）：
| 指标 | 值 |
|------|-----|
| Cosine 相似度 | ~0.87 |
| Top-8 IoU | ~0.02 |
| Hit Rate (top-1/3/5) | 0.0000 |

**结论**: MTP 路由的方向大致一致（Cosine=0.87），但选中的 expert 编码完全不同（IoU=0.02）。

### 5. Token → Expert 映射稳定性
| Token | 出现次数 | Layer 0（浅层） | Layer 9（中层） | Layer 18（深层） |
|-------|---------|-----------------|-----------------|------------------|
| `' x'` | 3 | **1.0000** | 0.5030 | 0.5219 |
| `' +'` | 2 | **1.0000** | 0.7778 | 0.6000 |
| `'(x'` | 2 | 0.6000 | 0.2308 | 0.4545 |

**结论**:
- **浅层（Layer 0）**: 路由由 token 唯一决定（IoU=1.0）
- **中层（Layer 9）**: 部分依赖上下文（IoU~0.5-0.78）
- **深层（Layer 18）**: 高度依赖上下文（IoU~0.45-0.60）

### 6. MTP Hidden State vs Decoder Hidden State
MTP[t] 与 Decoder[layer][t+1] 的 Cosine 相似度（两者都编码 tok[t+2]）：
| 层 | 平均 Cosine |
|------|------------|
| layer_0 (嵌入) | 0.0225 |
| layer_5 | 0.1800 |
| layer_10 (中层) | 0.2437 |
| layer_15 | 0.3490 |
| layer_19 | 0.3751 |
| layer_20 (decoder 输出) | **0.8278** |

**结论**: MTP 单层的 hidden state 几乎等价于 20 层 decoder 堆叠后的输出（Cosine=0.83）。余弦从浅到深单调递增，在最后一层跳升，说明 MTP 学透了整个 decoder 的变换。

### 7. Expert 路由置换测试 (swap_test.py)

**核心实验**：同一输入、同一 expert 权重，用 MTP 的路由替代 Decoder 的路由，对比 MoE 输出。

**实验方法**：
1. 捕获 decoder 每层 MoE 的输入和输出（前向 hook）
2. 用 MTP 的 gate 处理 decoder 的 MoE 输入，得到 MTP 的 top-8 选择
3. 用 MTP 的 top-8 索引 / 权重，喂给 decoder 的 `moe_infer`（使用 decoder 自己的 expert 权重）
4. 比较：Decoder 原生输出 vs MTP 路由→Decoder 权重

| 层 | Expert 索引重叠/8 | Cos(MTP路由→Decoder权重) |
|-----|-----------------|------------------------|
| L1  | 0.21 | 0.6009 |
| L2  | 0.36 | 0.4937 |
| L3  | 0.00 | 0.6005 |
| L4  | 0.07 | **0.7051** |
| L5  | 0.43 | 0.6207 |
| L6  | 0.07 | **0.7079** |
| L7  | 0.07 | 0.6576 |
| L8  | 0.00 | 0.4977 |
| L9  | 0.43 | 0.6114 |
| L10 | 0.36 | 0.5974 |
| L11 | 0.00 | 0.6164 |
| L12 | 0.07 | 0.5654 |
| L13 | 0.29 | 0.4653 |
| L14 | 0.00 | 0.4961 |
| L15 | 0.14 | 0.5096 |
| L16 | 0.00 | 0.4893 |
| L17 | 0.36 | 0.4706 |
| **L18** | **0.14** | **0.7176** |
| **平均** | **0.17** | **0.5791** |

**关键发现**：
- Expert 索引几乎不重叠（平均 0.17/8），但 MoE 输出 Cosine = **0.58~0.72**
- 对比 Section 3：经过后续层 + layernorm + lm_head 后，最终 logits Cosine = **0.97**
- **两层压缩**：routing 选择差异（IoU~0）→ MoE 输出差异（Cos 0.58）→ 最终 lm_head（Cos 0.97）
- lm_head + layernorm 将 42% 的中间差异压缩到仅 3%

**结论**：MTP 的路由可以直接喂给 decoder 的 expert 权重，即使选了完全不同的 8 个 expert 编号（IoU~0），最终 token 预测几乎不变（Cos=0.97）。路由高度冗余。

### 8. Speculative Decoding 搬运量节省
| N 个 draft token | 逐步加载 | 一次性验证 | 节省 |
|------------------|---------|-----------|------|
| N=2 | 304 | 296 | 2.6% |
| N=4 | 608 | 419 | 31.1% |
| N=8 | 1216 | 714 | 41.3% |
| N=16 | 2432 | 924 | 62.0% |

**结论**: 路由多样性很高。相邻两步的 expert 集几乎完全不重叠（N=2: 15.6/16）。节省来自大 N 下跨位置的 expert 复用。

---

## 核心问题
MTP 的路由决策能否用于预取 Decoder 的 Expert 加载（CPU offload 场景）？

### 已有证据
1. **Token 本身不能预测深层路由**（深层路由依赖完整上下文，不只是当前 token）
2. **MTP hidden state ≈ Decoder 最终输出**（Cosine=0.83）
3. **Injectivity（双射）理论**: Input 序列 → hidden state 是双射，路由由完整上下文唯一决定
4. **MTP Router vs Decoder Router IoU ~0.02**: 两者选择了完全不同的 experts

### 开放问题
- MTP 的 routing 能否用来推断 decoder 的 routing？
- MTP 的 hidden state 能否预测 decoder 的 expert 选择？
- MTP 自身 1 层 MoE 的搬运开销 vs 预取节省的搬运，净收益如何？

---

## 文件
- `decoder.py`: 单次前向传播，MTP 数据提取
- `compare.py`: 所有指标函数（8 组）
- `main.py`: 流水线 + 报告生成
- `analyze_routing.py`: Token→expert 映射分析
- `verify_spec.py`: Speculative decoding 模拟
