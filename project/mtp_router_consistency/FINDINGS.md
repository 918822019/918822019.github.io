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

## 核心矛盾

CPU offload 推理的瓶颈不是算力，是 PCIe 搬运。每层需要从 CPU→GPU 搬 8 个 expert 权重（256 选 8），18 层 × 8 = **144 次搬运**。算力大部分时间在等权重。

MTP 有一层自己的 MoE（256 experts），它在预测 token 时已经做了 routing。**能不能用它预测 decoder 每层需要的 expert，提前批量搬完？**

---

## 核心发现

### 前提：Expert 选择高度冗余

| 实验 | 指标 | 值 |
|------|------|-----|
| MTP 路由 → Decoder 权重 | MoE 输出 Cos | 0.58~0.72 |
| 同上，经 lm_head 后 | Logits Cos | **~0.97** |
| MTP 路由 vs Decoder 路由 | Expert 索引 IoU | **~0.02** |

**MTP 选了完全不同的 8 个 expert（IoU=0.02），但最终输出与 decoder 几乎一致（Cos=0.97）。** 这说明不存在"唯一正确的 8 个 expert"——不同的 8-expert 子集功能等价。

**这也意味着预测不需要精确**——猜错 expert 编号不影响最终结果。

### 数据总表

| # | 测试 | 关键数字 | 结论 |
|---|------|---------|------|
| 1 | Token 预测精度 | LM 25-75%, MTP 33-100% | MTP 看到 tok[t+1]嵌入，精度更高 |
| 2 | LM vs MTP logits | Cos=**0.97** | 两个独立 forward pass 输出几乎一致 |
| 3 | MTP hidden vs Decoder hidden | Cos=**0.83**（最后一层） | MTP 1 层 ≈ Decoder 20 层 |
| 4 | MTP 路由 → Decoder 权重 | MoE Cos=0.58, Logit Cos=**0.97** | **不同 expert 子集功能等价** |
| 5 | 跨 prompt 验证 | 深层 Logit Cos=0.78-0.92 | 深层预测稳定，浅层有 prompt 依赖 |
| 6 | 全层路由交换 | Token 匹配 7.1%, 坍缩到 `!` | 全层不可行，但单层/深层安全 |

---

## 逐层预取策略

利用两层互补方法：

### 浅层（L1-L10）：Token 查表

浅层路由由 token 唯一决定（L0 IoU=1.0，L9 IoU~0.5）。遇到重复 token 可以用历史 routing 记录直接查表。

### 深层（L11-L18）：MTP hidden → Gate 预测

MTP hidden state 编码了全部上下文，通过每层 decoder 的 gate 即可预测 routing（Cos=0.83~0.87，跨 prompt 稳定，std<0.05）。

```
MTP 前向:
  position t → MTP hidden state[t]（2048 维）
       ↓
  decoder[l].gate(hidden[t]) → routing logits（256 维）
       ↓
  top-8 experts → 批量预加载到 GPU

Decoder 前向:
  position t+1 → expert 已经在 GPU 上，零等待
```

### 高冗余 → 高容错

因为不同 expert 子集功能等价，MTP 预测即使不完全准确（IoU 0.17, Top-8 overlap 2.2/8），最终输出仍几乎不变（Cos=0.97）。这意味着：

1. **不需要完美预测**——猜错一半都能正常工作
2. **不需要 rejection sampling**——验证阶段自然兜底
3. **可以激进预取**——搬错 expert 不影响结果

---

## 关键测试详情

### 1. Expert 路由置换（swap_test.py）

同一输入、同一 expert 权重，替换 routing 前后对比：

| 层 | Expert IoU | MoE Cos |
|----|-----------|---------|
| L1 | 0.21/8 | 0.60 |
| L4 | 0.07/8 | 0.71 |
| L6 | 0.07/8 | 0.71 |
| L15 | 0.14/8 | 0.51 |
| L16 | 0.00/8 | 0.49 |
| L18 | 0.14/8 | 0.72 |
| **平均** | **0.17/8** | **0.58** |

**二层压缩**：expert 索引差异（IoU=0.17/8）→ MoE 输出差异（Cos=0.58）→ 经 lm_head 后（Cos=**0.97**）。中间层差异几乎被完全压缩。

### 2. MTP Hidden → Routing 预测（predict_test.py）

MTP hidden state 喂给每层 decoder gate，零样本预测 routing：

| 层 | Logits Cos | Top-8 Overlap |
|----|-----------|---------------|
| L1 | 0.74 | 1.7/8 |
| L3 | 0.83 | 1.3/8 |
| L6 | 0.78 | 3.4/8 |
| L13 | 0.85 | 2.0/8 |
| L15 | 0.87 | 3.0/8 |
| L16 | 0.88 | 2.6/8 |
| **平均** | **0.81** | **2.2/8** |

参考：MTP 自己 gate 预测自己 routing（上限）：Cos=0.95, Overlap=4.2/8

### 3. 跨 Prompt 稳定性（multi_test.py）

| Prompt | 类型 | Avg Cos | 深层(L13+) |
|--------|------|---------|------------|
| `def foo(x):...` | 代码 | 0.74 | 0.80-0.82 |
| `def fibonacci...` | 长代码 | 0.79 | 0.84-0.87 |
| `Machine learning...` | 技术文 | 0.68 | 0.91-0.92 |
| `The quick brown fox...` | 英文 | 0.09 | **0.78-0.87** |

**深层（L11-L18）跨 prompt 稳定**（std=0.03~0.1），浅层因 token 类型差异而不稳定。两种预取策略互补。

### 4. Speculative Decoding 搬运量节省

| N | 逐步加载 | 一次性验证 | 节省 |
|---|---------|-----------|------|
| 2 | 304 | 296 | 2.6% |
| 4 | 608 | 419 | 31.1% |
| 8 | 1216 | 714 | 41.3% |
| 16 | 2432 | 924 | **62.0%** |

---

## 文件

- `decoder.py` / `compare.py` / `main.py`: 分析管道
- `swap_test.py`: 路由置换测试
- `layer_test.py`: 分层交换测试
- `predict_test.py`: MTP hidden → routing 预测
- `multi_test.py`: 跨 prompt 验证
- `conf_test.py`: 置信度 + 语义分析
- `verify_spec.py`: Speculative decoding 模拟
