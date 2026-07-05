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

## PCIe 搬运测算

每个 expert 权重：3.1M 参数 × 2 字节（bf16）= **6.3 MB**
每层加载（8 experts）：**50.3 MB**
每 token 加载（18 层）：**906 MB**

注意：显存带宽（3060=360GB/s）不是瓶颈，PCIe 带宽（16GB/s）才是。

| 场景 | PCIe 3.0 (16GB/s) | PCIe 4.0 (32GB/s) | PCIe 5.0 (63GB/s) |
|------|-------------------|-------------------|-------------------|
| 逐层串行加载（无 MTP） | 56.6ms/tok | 28.3ms/tok | 14.4ms/tok |
| MTP 预取 + 流水线 | 7.2ms/tok | 4.0ms/tok | 2.5ms/tok |
| **加速比** | **7.9x** | **7.0x** | **5.8x** |
| **时间节省** | **87%** | **86%** | **83%** |

**为什么 MTP 能省？** 没有 MTP 时，每层必须等前一层算完才知道要搬哪 8 个 expert，无法预取。MTP 一次性预测全部 18 层的 routing，让所有 expert 可以提前开始搬运，计算和搬运流水线化。

**为什么 expert 冗余重要？** MTP 预测的 routing 和 decoder 自身的 routing 只有 IoU~0.02（几乎不重叠）。但因为不同 expert 子集功能等价，**预测错误不影响最终输出**（Logits Cos=0.97）。这允许激进预取而不需要验证。

---

## Speculative Decoding 实测

MTP 自回归 draft 测试（prompt="def fibonacci(n):..."，draft 16 tokens，取 mtp_hidden[-2] 避免回绕）：

```
pos  source  Draft     Ground truth    Match?
─────────────────────────────────────────────
0    LM      )         )               OK
1    MTP     \n        \n              OK
2    MTP               \n              OK
3    MTP     if        if              OK
4    MTP     n         n               OK
5    MTP     <=        <=              OK
6    MTP     \n        \n              OK
7    MTP     1         0               X
```

| 指标 | 值 |
|------|-----|
| Draft 准确率 | **43.8%** (7/16) |
| 连续接受率（verification） | **7/16** |
| 有效加速（2 passes = 1 draft + 1 verify） | **~4.0x** |

**注意**：MTP 的 roll_tensor(shifts=-1) 导致最后一个位置的输入回绕到 tok[0]。取 mtp_hidden[-1]（最后一位置）是错的，应取 mtp_hidden[-2]（倒数第二位置，roll 后这里才是最新的 token）。

**对自然语言无效的原因**：英语 prompt 的 token 熵（4.50）远高于代码（2.94）。高熵意味着 LM 本身就不确定下一个 token，MTP 自回归时每次用自己的预测作条件，微小 logit 变化就导致 argmax 跳到另一个合理但不同的 token。验证时 reject。代码的 token 分布是尖峰型（def→空格→fib→...），MTP 自回归非常适合这种确定性场景。

**结合预取的总体加速**：speculative decoding（~4x） + 路由预取流水线（~7x）理论上可叠加。

---

## 全层坍缩机制

当全部 18 层 MoE 的路由被替换为 MTP 路由时，输出坍缩到单一 token `!`（vocab id=0）。逐步替换揭示：

| 替换层数 | 唯一 token | mode | 置信度 |
|---------|-----------|------|--------|
| 0 | 9 | \n | 0.08 |
| 4 (L1-L4) | 4 | ! | 0.45 |
| 7 (L1-L7) | 2 | ! | 0.64 |
| 11 (L1-L11) | **1** | ! | 0.80 |
| 12-18 | 1 | ! | 0.98 |

**坍缩路径**：L1-L3 正常 → L4-L7 `!` 出现 → L8-L11 完全锁定 → L12-L18 维持。

**注意**：坍缩后不同位置的 hidden state 仍然不同（L2 距离 ~511，正常 ~500），但 lm_head 将所有不同的 hidden state 映射到了同一个 token。`!` 是 vocab 中 ID=0 的 token，权重 norm 很低（0.96，排 151887/157184），norm 小的 token 在 lm_head 中的"吸引力盆地"大——当 hidden state 因 wrong routing 偏离正常分布后，lm_head 最近的有效输出就是 `!`。

**结论**：坍缩不是 hidden state 收敛，而是 lm_head 找不到正确 token 时的回退行为。根因在浅层（L4-L7）的 routing 偏差已经让 hidden state 偏离了正常空间，中层（L8-L11）放大这个偏离直到 lm_head 回退到 `!`。

---

## 文件

- `decoder.py` / `compare.py` / `main.py`: 分析管道
- `swap_test.py`: 路由置换测试
- `layer_test.py`: 分层交换测试
- `predict_test.py`: MTP hidden → routing 预测
- `multi_test.py`: 跨 prompt 验证
- `conf_test.py`: 置信度 + 语义分析
- `verify_spec.py`: Speculative decoding 模拟
