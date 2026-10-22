# MTP 多维度分析报告

**模型**: data/models/Ling-mini-base-2.0
**设备**: cpu
**分析模式**: 单次前向传播 (不生成新 token)
**用时**: 16.1s
**测试样本数**: 2

## 0) Router 一致性指标

| 指标 | 均值 | 最小值 | 最大值 |
|------|------|--------|--------|
| cosine_sim | 0.8600 | 0.8536 | 0.8664 |
| js_div | 0.5646 | 0.5617 | 0.5676 |
| kl_div | 5.1847 | 5.1299 | 5.2394 |
| spearman_rho | 0.0029 | -0.0277 | 0.0335 |
| top_1_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_3_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_5_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_8_iou | 0.0083 | 0.0000 | 0.0167 |

## 1) Router 置信度 / 熵分析

| 指标 | 含义 | 值 |
|------|------|-----|
### 样本 1
| MTP Router 熵（越高越不确定） | 3.1585 |
| Decoder Router 熵 | 2.6934 |
| 熵差 (MTP - Decoder) | 0.4651 |
| MTP Router 置信度 (top-1 prob) | 0.2436 |
| Decoder Router 置信度 | 0.3472 |
| 置信度差 (MTP - Decoder) | -0.1036 |

### 样本 2
| MTP Router 熵（越高越不确定） | 2.9610 |
| Decoder Router 熵 | 3.0472 |
| 熵差 (MTP - Decoder) | -0.0862 |
| MTP Router 置信度 (top-1 prob) | 0.2386 |
| Decoder Router 置信度 | 0.3089 |
| 置信度差 (MTP - Decoder) | -0.0703 |

## 2) Expert 命中分析

| 指标 | 含义 | 值 |
|------|------|-----|
### 样本 1
| 平均重叠 expert 数 (top-k 中) | 0.2500 |
| 最大重叠 expert 数 | 1.0000 |
| 最小重叠 expert 数 | 0.0000 |
| 重叠比例 (overlap/k) | 0.0312 |
| 零重叠比例 (完全不命中) | 0.7500 |
| 完全重叠比例 (k/k) | 0.0000 |

### 样本 2
| 平均重叠 expert 数 (top-k 中) | 0.0000 |
| 最大重叠 expert 数 | 0.0000 |
| 最小重叠 expert 数 | 0.0000 |
| 重叠比例 (overlap/k) | 0.0000 |
| 零重叠比例 (完全不命中) | 1.0000 |
| 完全重叠比例 (k/k) | 0.0000 |

## 3) 逐层 Router 对比 (MTP vs 各 Decoder Layer)

| 指标 | 值 |
|------|-----|
### 样本 1
| 逐层 Cosine 均值 | 0.8875 |
| 逐层 Top-K IoU 均值 | 0.0188 |
| 最后一层 Cosine (即原指标) | 0.8599 |
| 最后一层 Top-K IoU | 0.0167 |
| 末层 Cosine - 首层 Cosine | -0.0389 |

逐层 Cosine 序列 (浅→深):

`0.899, 0.923, 0.902 ... 0.876, 0.875, 0.860`

### 样本 2
| 逐层 Cosine 均值 | 0.8855 |
| 逐层 Top-K IoU 均值 | 0.0168 |
| 最后一层 Cosine (即原指标) | 0.8711 |
| 最后一层 Top-K IoU | 0.0167 |
| 末层 Cosine - 首层 Cosine | -0.0283 |

逐层 Cosine 序列 (浅→深):

`0.899, 0.913, 0.901 ... 0.871, 0.873, 0.871`

## 4) 输出 logits 关联分析 (LM Head vs MTP Router)

| 指标 | 含义 | 值 |
|------|------|-----|
### 样本 1
| LM 置信度 vs MTP 置信度 相关系数 | 0.2794 |
| LM Head 平均置信度 | 0.5045 |
| LM Head 置信度标准差 | 0.1481 |

### 样本 2
| LM 置信度 vs MTP 置信度 相关系数 | -0.8415 |
| LM Head 平均置信度 | 0.2868 |
| LM Head 置信度标准差 | 0.2311 |

## 5) 位置趋势分析

### 样本 1
| 指标 | 前半均值 | 后半均值 | 趋势 |
|------|----------|----------|------|
| kl_div | 4.6341 | 5.8447 | up |
| js_div | 0.5325 | 0.5908 | up |
| cosine_sim | 0.8431 | 0.8641 | up |
| top_8_iou | 0.0000 | 0.0333 | down |
| spearman_rho | -0.0214 | -0.0341 | up |
| top_1_hit_rate | 0.0000 | 0.0000 | down |
| top_3_hit_rate | 0.0000 | 0.0000 | down |
| top_5_hit_rate | 0.0000 | 0.0000 | down |

### 样本 2
| 指标 | 前半均值 | 后半均值 | 趋势 |
|------|----------|----------|------|
| kl_div | 4.3476 | 5.9122 | up |
| js_div | 0.5435 | 0.5916 | up |
| cosine_sim | 0.8547 | 0.8781 | up |
| top_8_iou | 0.0000 | 0.0000 | down |
| spearman_rho | 0.0358 | 0.0311 | up |
| top_1_hit_rate | 0.0000 | 0.0000 | down |
| top_3_hit_rate | 0.0000 | 0.0000 | down |
| top_5_hit_rate | 0.0000 | 0.0000 | down |

## 6) Token 贪心解码准确率

比较 Decoder (LM Head) 和 MTP Head 在贪心解码下预测的 token 是否一致。

| 样本 | 准确率 | 正确数/总数 |
|------|--------|-------------|
| 样本 1 | 100.00% | 4/4 |
| 样本 2 | 50.00% | 2/4 |

## 7) LM head vs MTP head 完整 Logits 分布对比

对齐目标： `lm_logits[:, t+1, :]` vs `mtp_logits[:, t, :]` 都预测 `token[t+2]`

| 指标 | 含义 | 样本1 | 样本2 |
|------|------|-------|-------|
| lm_mtp_logit_cosine | Cosine 相似度 | 0.9761 | 0.9799 |
| lm_mtp_logit_kl | KL 散度 | 0.3550 | 0.6203 |
| lm_mtp_logit_js | JS 散度 | 0.0866 | 0.1341 |
| lm_mtp_logit_spearman | Spearman 相关 | 0.8447 | 0.8668 |
| lm_mtp_logit_top8_iou | Top-8 IoU | 0.5111 | 0.3587 |
| lm_mtp_logit_prob_dot | 概率分布点积 | 0.2278 | 0.0670 |
| lm_mtp_logit_l2 | Logits L2 距离 | 509.1209 | 476.9328 |

**解读**: Cosine ~1.0 且 KL/JS ≈ 0 说明两个 logits 分布几乎相同，与 injectivity 理论一致。

## 8) 对真实 Token (Ground Truth) 的预测精度

比较 LM Head / MTP Head 的贪心解码结果与序列中的真实 token。

| 样本 | LM→tok[t+1] 精度 | MTP→tok[t+2] 精度 | MTP-LM 差距 |
|------|-------------------|-------------------|-------------|
| 样本 1 | 75.00% (3/4) | 100.00% (3/3) | +25.00% |
| 样本 2 | 25.00% (1/4) | 33.33% (1/3) | +8.33% |

**解读**: LM 精度反映了模型对 prompt 内已知 token 的拟合程度; MTP 精度反映其在看到 tok[t+1] 嵌入后预测 tok[t+2] 的能力。

## 各样本详情

### 样本 1

**Prompt**: `def fibonacci(n):`
**可比较位置数**: 4

| 路由指标 | 值 |
|----------|-----|
| kl_div | 5.2394 |
| js_div | 0.5617 |
| cosine_sim | 0.8536 |
| top_8_iou | 0.0167 |
| spearman_rho | -0.0277 |
| top_1_hit_rate | 0.0000 |
| top_3_hit_rate | 0.0000 |
| top_5_hit_rate | 0.0000 |

### 样本 2

**Prompt**: `def merge_sort(arr):`
**可比较位置数**: 4

| 路由指标 | 值 |
|----------|-----|
| kl_div | 5.1299 |
| js_div | 0.5676 |
| cosine_sim | 0.8664 |
| top_8_iou | 0.0000 |
| spearman_rho | 0.0335 |
| top_1_hit_rate | 0.0000 |
| top_3_hit_rate | 0.0000 |
| top_5_hit_rate | 0.0000 |
