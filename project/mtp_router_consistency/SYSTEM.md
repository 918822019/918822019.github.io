# MTP Router 一致性分析 - 系统文档

## 概述

### 背景
BailingMoeV2 (12B, 256 experts, 18 MoE decoder layers, top-8 routing) 在 6GB VRAM 上推理受 CPU offload PCIe 瓶颈限制。

### 瓶颈
| 指标 | 数值 |
|------|------|
| 显存带宽 (RTX 3060) | 360 GB/s |
| PCIe 3.0 x16 带宽 | 16 GB/s |
| 每层加载 8 experts (bf16) | 50.3 MB |
| 每 token 加载 18 层 | 906 MB |
| PCIe 传输时间 (串行) | 56.6 ms |
| GPU 计算时间 | < 1 ms |

### 核心问题
MTP 有一层自己的 MoE（256 experts），它在预测 token 时已经做了 routing。能否用它预测 decoder 每层需要的 expert，提前批量搬完，消除 PCIe 等待？

---

## 目录

1. [MTP 架构](#1-mtp-架构)
2. [Expert 冗余性](#2-expert-冗余性)
3. [路由预测](#3-路由预测)
4. [Speculative Decoding](#4-speculative-decoding)
5. [全层坍缩机制](#5-全层坍缩机制)
6. [分层交换测试](#6-分层交换测试)
7. [PCIe 加速方案](#7-pcie-加速方案)
8. [文件索引](#8-文件索引)

---

## 1. MTP 架构

### 1.1 层结构
- Decoder: 20 层（第 0 层为 dense MLP，第 1-18 层为 MoE，第 19 层为 MTP）
- MoE 每层 256 experts，top-8 activated
- MTP 层拥有**自己独立的 256 experts**，不与 decoder 共享

### 1.2 MTP 预测路径
```
MTP Layer:
  [shifted tok[t+1] emb, decoder_hidden[t]]
  → eh_proj → Self-Attention → MoE (256 experts) → lm_head → tok[t+2]
```

### 1.3 关键特性
- MTP 通过内部 `roll_tensor(shifts=-1)` 看到 tok[t+1] 的嵌入，比 LM 多一个信息优势
- MTP 预测 tok[t+2]，LM 预测 tok[t+1]
- **对齐**：`lm_logits[:, t+1, :]` vs `mtp_logits[:, t, :]` 都预测同一位 token[t+2]

### 1.4 Token 预测精度

| 指标 | Code prompt | English prompt |
|------|------------|----------------|
| LM→tok[t+1] (ground truth) | 25-75% | — |
| MTP→tok[t+2] (ground truth) | 33-100% | — |
| LM vs MTP argmax 一致性 | **100%** | **~55%** |
| 平均 token 熵 | **2.94** | **4.50** |

> 代码 token 分布呈尖峰型（低熵），英语呈扁平型（高熵）。高熵导致 MTP 自回归时微小 logit 变化即改变 argmax。

---

## 2. Expert 冗余性

### 2.1 核心实验

**方法**：同一输入、同一 expert 权重，用 MTP 路由替代 Decoder 路由，对比 MoE 输出。

| 层 | Expert 索引 IoU | MoE 输出 Cos |
|----|----------------|-------------|
| L1 | 0.21/8 | 0.60 |
| L4 | 0.07/8 | 0.71 |
| L6 | 0.07/8 | 0.71 |
| L10 | 0.36/8 | 0.60 |
| L15 | 0.14/8 | 0.51 |
| L18 | 0.14/8 | 0.72 |
| **平均** | **0.17/8** | **0.58** |

### 2.2 两层压缩路径

```
Expert 索引差异 (IoU=0.17/8)
  → MoE 输出 Cos=0.58
    → lm_head 后 Logits Cos=**0.97**
```

### 2.3 结论

**MTP 选了完全不同的 8 个 expert（IoU=0.02），但最终输出几乎一致（Cos=0.97）。** 不同 8-expert 子集功能等价。预测不需要精确——猜错 expert 编号不影响最终结果。

---

## 3. 路由预测

### 3.1 MTP Hidden → Decoder Gate

MTP hidden state（2048 维）编码了全部上下文。直接喂给每层 decoder 的 gate（线性层 2048→256），零样本预测 routing。

| 层 | Routing Logits Cos | Top-8 Overlap |
|----|-------------------|---------------|
| L1 | 0.74 | 1.7/8 |
| L6 | 0.78 | 3.4/8 |
| L13 | 0.85 | 2.0/8 |
| L15 | 0.87 | 3.0/8 |
| L16 | **0.88** | 2.6/8 |
| **平均** | **0.81** | **2.2/8** |
| 参考 (MTP self) | 0.95 | 4.2/8 |

### 3.2 跨 Prompt 稳定性

| Prompt 类型 | 整体 Avg Cos | 深层(L13+) Cos | 稳定性 |
|------------|-------------|----------------|--------|
| 代码 | 0.74-0.79 | 0.80-0.90 | 稳定 |
| 技术文 | 0.68 | 0.91-0.92 | 稳定 |
| 英文 | **0.09** | **0.78-0.87** | **深层稳定** |

**关键发现**：
- **浅层（L1-L10）**：路由预测不稳定（跨 prompt std=0.3~0.5），英文下负相关。浅层路由由 token 决定
- **深层（L11-L18）**：路由预测跨 prompt 稳定（std=0.03~0.1），所有类型 Cos≥0.78。深层路由由语义决定

### 3.3 预取流程

```
MTP 前向:
  position t → MTP hidden state[t]（2048 维）
       ↓
  decoder[l].gate(hidden[t]) → routing logits（256 维）
       ↓
  top-8 experts → 批量预加载到 GPU

Decoder 前向:
  position t+1 → expert 已在 GPU 上，零等待
```

---

## 4. Speculative Decoding

### 4.1 MTP 自回归 Draft

MTP 一次预测一个 token（tok[t+2]）。要预测多个 token，必须自回归：

```
Step 1: LM 生成 tok[T]（✓）
Step 2: MTP sees tok[T] → 预测 tok[T+1]
Step 3: MTP sees 自己的 tok[T+1] → 预测 tok[T+2]
...
```

**⚠️ 关键实现细节**：MTP 内部 `roll_tensor(shifts=-1)` 导致最后一个位置的输入回绕到 tok[0]。应取 `mtp_hidden[-2]` 而非 `mtp_hidden[-1]`。

### 4.2 代码 Prompt 结果

```
pos  source  Draft     Ground truth   Accept?
──────────────────────────────────────────────
0    LM      )         )              OK
1    MTP     \n        \n             OK
2    MTP               \n             OK
3    MTP     if        if             OK
4    MTP     n         n              OK
5    MTP     <=        <=             OK
6    MTP     \n        \n             OK
7    MTP     1         0              X
```

| 指标 | 值 |
|------|-----|
| Draft 准确率 | 43.8% (7/16) |
| 连续接受率 | 7/16 |
| 有效加速（2 pass: 1 draft + 1 verify） | **~4.0x** |

### 4.3 English Prompt 结果

| Prompt | 类型 | 接受率 | 加速 |
|--------|------|--------|------|
| fibonacci | 代码 | 3/4 (75%) | 2.0x |
| merge_sort | 代码 | 4/4 (100%) | 2.5x |
| english | 自然语言 | 1/4 (25%) | **1.0x (无效)** |

**原因**：英语 token 熵 4.50 vs 代码 2.94。高熵时 MTP 自回归的微小 logit 偏移即改变 argmax。

### 4.4 结合预取

Speculative decoding（~4x） + 路由预取流水线（~7x）理论上可叠加至 **~28x** 总加速（代码场景）。

---

## 5. 全层坍缩机制

当全部 18 层 MoE 的路由替换为 MTP 路由时，输出坍缩到单一 token `!`（vocab id=0）。

### 5.1 逐步坍缩

| 替换层数 | 唯一 token | 多数 token | 置信度 |
|---------|-----------|-----------|--------|
| 0 | 9 | \n | 0.08 |
| 4 (L1-L4) | 4 | ! | 0.45 |
| 7 (L1-L7) | 2 | ! | 0.64 |
| 11 (L1-L11) | **1** | ! | 0.80 |
| 12-18 | 1 | ! | 0.98 |

### 5.2 坍缩真相

- 坍缩后不同位置的 hidden state **仍然不同**（L2 距离 ~511，正常 ~500）
- 但 lm_head 将所有不同 hidden state 映射到同一 token
- `!`（token id=0）权重 norm=0.96（排 151887/157184），"吸引力盆地"最大
- hidden state 偏离正常空间 → lm_head 找不到正确 token → 回退到 `!`

### 5.3 坍缩路径

```
L1-L3: 正常
L4-L7: ! 开始出现（routing 偏差让 hidden state 偏离正常空间）
L8-L11: 完全锁定（中层放大偏离）
L12-L18: 维持坍缩
```

### 5.4 对方案的意义

- **浅层单独换没问题**：hidden state 仍在正常分布内
- **深层单独换没问题**：先经过浅层正常 routing 处理，深层偏离被 lm_head 吸收
- **中层（L5-L13）+ 浅层一起换 → 坍缩**：浅层偏离 + 中层放大 → lm_head 回退

---

## 6. 分层交换测试

### 6.1 单组交换

| 配置 | Token 匹配率 | LogitCos | 坍缩? |
|------|-------------|----------|-------|
| **Deep (L14-L18)** | **92.9%** | 0.9903 | 否 |
| Single L9 | 100.0% | 0.9996 | 否 |
| Single L18 | 85.7% | 0.9987 | 否 |
| Shallow (L1-L4) | 57.1% | 0.9775 | 否 |
| Shallow+Deep | 42.9% | 0.9764 | 否 |
| Middle (L5-L13) | **14.3%** | 0.9795 | 否 |
| ALL 18 层 | **7.1%** | 0.9579 | **是** |

### 6.2 结论

- 深层可安全替换（92.9%），中层是误差放大器（14.3%）
- LogitCos 与 TokenMatch 解耦：所有配置 LogitCos > 0.95，但 TokenMatch 从 7% 到 100%
- 全层替换导致坍缩，但分层替换安全

### 6.3 建议方案

| 层范围 | 策略 | 方法 | 覆盖 |
|-------|------|------|------|
| L1-L10 | Token 查表 | 浅层路由由 token 唯一决定（IoU~1.0） | 10/18 层 |
| L11-L18 | MTP hidden → Gate | 跨 prompt 稳定预测（Cos=0.83~0.87） | 8/18 层 |
| **合计** | **混合策略** | 两层互补 | **100%** |

---

## 7. PCIe 加速方案

### 7.1 数据量

| 项目 | 数值 |
|------|------|
| 每个 expert (bf16) | 6.3 MB |
| 每层 8 experts | 50.3 MB |
| 全部 18 层 | **906 MB/token** |
| 全部 256 experts（常驻 GPU 不可能） | 29 GB > 6 GB VRAM |

### 7.2 加速对比

| 场景 | PCIe 3.0 | PCIe 4.0 | PCIe 5.0 |
|------|---------|---------|---------|
| 串行（无 MTP） | 56.6 ms | 28.3 ms | 14.4 ms |
| MTP 预取 + 流水线 | **7.2 ms** | **4.0 ms** | **2.5 ms** |
| 加速比 | **7.9x** | **7.0x** | **5.8x** |

### 7.3 加速来源

- **MTP 一次性预测全部 18 层 routing** → 开启异步预取
- 无 MTP：每层需等前一层 gate 算完 → 串行加载
- 有 MTP：所有 routing 提前知道 → 加载与计算流水线化
- **expert 冗余**：搬错 expert 不影响输出（Logit Cos=0.97），允许激进预取

### 7.4 总加速（代码场景）

| 技术 | 单独加速 | 叠加 | 备注 |
|------|---------|------|------|
| 预取流水线 | 7.9x | — | 消除 PCIe 等待 |
| Spec Decoding | 4.0x | — | 减少 forward pass 次数 |
| **叠加** | **~28x** | 理论值 | 实际受 SD 接受率影响 |

---

## 8. 文件索引

| 文件 | 类型 | 用途 |
|------|------|------|
| `decoder.py` | .py | 单次前向 MTP 数据提取 |
| `compare.py` | .py | 8 组 metric 函数 |
| `main.py` | .py | 流水线 + 报告生成 |
| `model_utils.py` | .py | 模型加载工具 |
| `config.py` | .py | 配置参数 |
| `swap_test.ipynb` | notebook | 路由置换测试（MTP 路由→Decoder 权重） |
| `layer_test.ipynb` | notebook | 分层交换测试（确定安全交换范围） |
| `predict_test.ipynb` | notebook | MTP hidden → Decoder gate 零样本路由预测 |
| `multi_test.ipynb` | notebook | 跨 prompt 验证（5 种 prompt 类型） |
| `conf_test.ipynb` | notebook | 置信度 + 语义分析 |
| `full_test.ipynb` | notebook | 全层路由交换 end-to-end |
| `collapse_test.ipynb` | notebook | 全层坍缩机制分析 |
| `draft_test.ipynb` | notebook | MTP autoregressive drafting |
| `sd_multi.ipynb` | notebook | 跨 prompt spec decoding 对比 |
| `verify_spec.ipynb` | notebook | Speculative decoding 搬运量模拟 |
| `analyze_routing.ipynb` | notebook | Token→expert 映射分析 |

---

*最后更新: 2026-07-06*
