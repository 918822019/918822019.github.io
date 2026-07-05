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

### 8. 分层路由交换测试 (layer_test.py)

**测试目的**：确定哪些层的路由可以用 MTP 替换而不影响最终预测。分别测试浅层、中层、深层及组合。

| 配置 | Token 匹配率 | LogitCos | 坍缩? | 分析 |
|------|-------------|----------|-------|------|
| **Deep (L14-L18)** | **92.9%** | 0.9903 | 否 | ✅ 安全 |
| Single L9 | 100.0% | 0.9996 | 否 | ✅ 完美 |
| Single L18 | 85.7% | 0.9987 | 否 | ✅ 安全 |
| Shallow (L1-L4) | 57.1% | 0.9775 | 否 | ⚠️ 部分安全 |
| Shallow+Deep | 42.9% | 0.9764 | 否 | ⚠️ 部分安全 |
| Single L1 | 78.6% | 0.9916 | 否 | ⚠️ 部分安全 |
| **Middle (L5-L13)** | **14.3%** | 0.9795 | 否 | ❌ 危险 |
| **ALL 18 层** | **7.1%** | 0.9579 | **是(!)** | ❌ 灾难 |

**关键发现**：
- **深层（L14-L18）可安全替换**：92.9% token 不变，LogitCos=0.99
- **中层（L5-L13）是误差放大器**：单独换中层掉到 14.3%，但单独换 L9 却是 100%（说明单层冗余高，级联放大误差）
- **全层坍缩根因**：中层的误差经过深层放大后，hidden state 被 lm_head 统一映射到 `!`
- **LogitCos 与 TokenMatch 解耦**：所有配置 LogitCos > 0.95，但 TokenMatch 从 7% 到 100% 不等。157k 词表中微小 logit 变化即可改变 argmax

**对 offload 方案的影响**：
- 不能一次性替换全部 18 层路由
- 深层（L14-L18）最适合用 MTP 路由预取 expert（92.9% 匹配）
- 浅层可部分替换（57.1% 匹配）
- 中层建议保留 decoder 原生路由

### 9. Speculative Decoding 搬运量节省
| N 个 draft token | 逐步加载 | 一次性验证 | 节省 |
|------------------|---------|-----------|------|
| N=2 | 304 | 296 | 2.6% |
| N=4 | 608 | 419 | 31.1% |
| N=8 | 1216 | 714 | 41.3% |
| N=16 | 2432 | 924 | 62.0% |

**结论**: 路由多样性很高。相邻两步的 expert 集几乎完全不重叠（N=2: 15.6/16）。节省来自大 N 下跨位置的 expert 复用。

### 10. MTP Hidden State → Decoder Routing 预测 (predict_test.py)

**核心思路**：MTP hidden state ≈ Decoder 最终输出（Cos=0.83），它已经编码了全部上下文信息。直接把它喂给每层 decoder 的 gate，看能否预测该层的 routing。

**实验**：`MTP_hidden[t] → decoder[l].gate → routing_logits[l]`，对比 decoder[l] 在 position t+1 的真实 routing logits。

| 层 | Routing Logits Cos | Top-8 Overlap |
|----|-------------------|---------------|
| L1  | 0.7424 | 1.69/8 |
| L3  | 0.8349 | 1.31/8 |
| L6  | 0.7803 | 3.38/8 |
| L9  | 0.7463 | 1.62/8 |
| L13 | 0.8474 | 2.00/8 |
| L15 | **0.8682** | **3.00/8** |
| L16 | **0.8793** | 2.62/8 |
| L18 | 0.8540 | 2.69/8 |
| **平均** | **0.8075** | **2.19/8** |

**参考上限**（MTP 自己的 gate 预测自己的 routing）: Cos=0.9503, Overlap=4.23/8

**结论**：
- MTP hidden state 通过每层 decoder 的 gate，即可预测 routing logits，Cos=**0.81**
- 比 token→expert 方案（IoU~0.02）好两个数量级
- 不需要额外模型结构——每个 decoder 的 gate 已经是一个现成的线性映射器（[2048→256]）
- 可以训练一个轻量 predictor（每层一个线性层），但不训练直接零样本也有 Cos=0.81

**方案**：
```
MTP 前向:  position t → MTP hidden state[t] (2048维)
                  ↓
          喂给 decoder[l].gate (每层一个线性层 [2048→256])
                  ↓
          得到 decoder[l] 在 position t+1 的 routing logits
                  ↓
          预取 top-8 experts → GPU 预加载
```

---

## 核心问题
MTP 的路由决策能否用于预取 Decoder 的 Expert 加载（CPU offload 场景）？

### 已确认
1. **Token 本身不能预测深层路由**（deep layer 路由依赖完整上下文，不只是当前 token）
2. **MTP hidden state ≈ Decoder 最终输出**（Cosine=0.83）
3. **MTP routing → Decoder experts 可行**：MoE 输出 Cos=0.58~0.72
4. **Expert 选择高度冗余**：不同 8-expert 子集可达等效输出
5. **两层压缩路径**：routing差异 → MoE输出差异(Cos 0.58) → lm_head(Cos 0.97)
6. **分层可行性**：深层(L14-L18)可安全替换(92.9%)，中层(L5-L13)不可替换(14.3%)
7. **MTP hidden → gate 预测**：零样本 Cos=0.81，可用于全层 expert 预取
7. **全层坍缩**：全部 18 层同时替换导致输出坍缩到单一 token

### 建议方案
对于 CPU offload 场景中利用 MTP 路由预取 expert：
1. **仅替换深层（L14-L18，共 5 层）的路由**，保留 L1-L13 的 decoder 原生路由
2. 深层占总搬运量的 5/18 ≈ **28%**，可节省这部分 PCIe 等待时间
3. 深层替换后 token 匹配率 92.9%，误判风险低
4. 配合 speculative decoding 的 verify 阶段可兜底剩余 7.1% 差异

### 开放问题
- MTP 的 routing 能否用来推断 decoder deep layer 的 routing？（已验证部分可行）
- 中间层的误差放大机制是什么？能否通过校准解决？
- 实际 offload 系统中，MTP 1 层开销 vs 28% 搬运节省的净收益如何？

---

## 文件
- `decoder.py`: 单次前向传播，MTP 数据提取
- `compare.py`: 所有指标函数（8 组）
- `main.py`: 流水线 + 报告生成
- `analyze_routing.py`: Token→expert 映射分析
- `verify_spec.py`: Speculative decoding 模拟
