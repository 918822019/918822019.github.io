# MTP 多维度分析报告

**模型**: data/models/Ling-mini-base-2.0
**设备**: cpu
**分析模式**: 单次前向传播 (不生成新 token)
**用时**: 13.8s
**测试样本数**: 2

## 0) Router 一致性指标

| 指标 | 均值 | 最小值 | 最大值 |
|------|------|--------|--------|
| cosine_sim | 0.8690 | 0.8639 | 0.8741 |
| js_div | 0.6109 | 0.5948 | 0.6270 |
| kl_div | 6.2406 | 6.0272 | 6.4540 |
| spearman_rho | 0.0176 | -0.0028 | 0.0380 |
| top_1_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_3_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_5_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_8_iou | 0.0167 | 0.0000 | 0.0333 |

## 1) Router 置信度 / 熵分析

| 指标 | 含义 | 值 |
|------|------|-----|
### 样本 1
| MTP Router 熵（越高越不确定） | 2.9570 |
| Decoder Router 熵 | 2.3214 |
| 熵差 (MTP - Decoder) | 0.6356 |
| MTP Router 置信度 (top-1 prob) | 0.2493 |
| Decoder Router 置信度 | 0.4169 |
| 置信度差 (MTP - Decoder) | -0.1676 |

### 样本 2
| MTP Router 熵（越高越不确定） | 2.6665 |
| Decoder Router 熵 | 2.0965 |
| 熵差 (MTP - Decoder) | 0.5700 |
| MTP Router 置信度 (top-1 prob) | 0.2615 |
| Decoder Router 置信度 | 0.5326 |
| 置信度差 (MTP - Decoder) | -0.2711 |

## 2) Expert 命中分析

| 指标 | 含义 | 值 |
|------|------|-----|
### 样本 1
| 平均重叠 expert 数 (top-k 中) | 0.5000 |
| 最大重叠 expert 数 | 1.0000 |
| 最小重叠 expert 数 | 0.0000 |
| 重叠比例 (overlap/k) | 0.0625 |
| 零重叠比例 (完全不命中) | 0.5000 |
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
| 逐层 Cosine 均值 | 0.8954 |
| 逐层 Top-K IoU 均值 | 0.0169 |
| 最后一层 Cosine (即原指标) | 0.8667 |
| 最后一层 Top-K IoU | 0.0167 |
| 末层 Cosine - 首层 Cosine | -0.0359 |

逐层 Cosine 序列 (浅→深):

`0.903, 0.924, 0.907 ... 0.884, 0.881, 0.867`

### 样本 2
| 逐层 Cosine 均值 | 0.8947 |
| 逐层 Top-K IoU 均值 | 0.0185 |
| 最后一层 Cosine (即原指标) | 0.8769 |
| 最后一层 Top-K IoU | 0.0333 |
| 末层 Cosine - 首层 Cosine | -0.0246 |

逐层 Cosine 序列 (浅→深):

`0.901, 0.918, 0.895 ... 0.885, 0.882, 0.877`

## 4) 输出 logits 关联分析 (LM Head vs MTP Router)

| 指标 | 含义 | 值 |
|------|------|-----|
### 样本 1
| LM 置信度 vs MTP 置信度 相关系数 | 0.9428 |
| LM Head 平均置信度 | 0.6157 |
| LM Head 置信度标准差 | 0.1187 |

### 样本 2
| LM 置信度 vs MTP 置信度 相关系数 | 0.0205 |
| LM Head 平均置信度 | 0.5981 |
| LM Head 置信度标准差 | 0.2971 |

## 5) 位置趋势分析

### 样本 1
| 指标 | 前半均值 | 后半均值 | 趋势 |
|------|----------|----------|------|
| kl_div | 5.4259 | 6.6286 | up |
| js_div | 0.5683 | 0.6212 | up |
| cosine_sim | 0.8459 | 0.8819 | up |
| top_8_iou | 0.0333 | 0.0333 | down |
| spearman_rho | -0.0165 | 0.0110 | up |
| top_1_hit_rate | 0.0000 | 0.0000 | down |
| top_3_hit_rate | 0.0000 | 0.0000 | down |
| top_5_hit_rate | 0.0000 | 0.0000 | down |

### 样本 2
| 指标 | 前半均值 | 后半均值 | 趋势 |
|------|----------|----------|------|
| kl_div | 5.9072 | 7.0007 | up |
| js_div | 0.6013 | 0.6528 | up |
| cosine_sim | 0.8545 | 0.8936 | up |
| top_8_iou | 0.0000 | 0.0000 | down |
| spearman_rho | 0.0470 | 0.0290 | up |
| top_1_hit_rate | 0.0000 | 0.0000 | down |
| top_3_hit_rate | 0.0000 | 0.0000 | down |
| top_5_hit_rate | 0.0000 | 0.0000 | down |

## 6) Token 贪心解码准确率

比较 Decoder (LM Head) 和 MTP Head 在贪心解码下预测的 token 是否一致。

| 样本 | 准确率 | 正确数/总数 |
|------|--------|-------------|
| 样本 1 | 100.00% | 4/4 |
| 样本 2 | 100.00% | 4/4 |

## 7) LM head vs MTP head 完整 Logits 分布对比

对齐目标： `lm_logits[:, t+1, :]` vs `mtp_logits[:, t, :]` 都预测 `token[t+2]`

| 指标 | 含义 | 样本1 | 样本2 |
|------|------|-------|-------|
| lm_mtp_logit_cosine | Cosine 相似度 | 0.9715 | 0.9650 |
| lm_mtp_logit_kl | KL 散度 | 0.3144 | 0.4777 |
| lm_mtp_logit_js | JS 散度 | 0.0833 | 0.1050 |
| lm_mtp_logit_spearman | Spearman 相关 | 0.8342 | 0.7867 |
| lm_mtp_logit_top8_iou | Top-8 IoU | 0.5362 | 0.5111 |
| lm_mtp_logit_prob_dot | 概率分布点积 | 0.3631 | 0.4414 |
| lm_mtp_logit_l2 | Logits L2 距离 | 508.0458 | 626.6219 |

**解读**: Cosine ~1.0 且 KL/JS ≈ 0 说明两个 logits 分布几乎相同，与 injectivity 理论一致。

## 各样本详情

### 样本 1

**Prompt**: `def fibonacci(n):`
**可比较位置数**: 4

| 路由指标 | 值 |
|----------|-----|
| kl_div | 6.0272 |
| js_div | 0.5948 |
| cosine_sim | 0.8639 |
| top_8_iou | 0.0333 |
| spearman_rho | -0.0028 |
| top_1_hit_rate | 0.0000 |
| top_3_hit_rate | 0.0000 |
| top_5_hit_rate | 0.0000 |

### 样本 2

**Prompt**: `def merge_sort(arr):`
**可比较位置数**: 4

| 路由指标 | 值 |
|----------|-----|
| kl_div | 6.4540 |
| js_div | 0.6270 |
| cosine_sim | 0.8741 |
| top_8_iou | 0.0000 |
| spearman_rho | 0.0380 |
| top_1_hit_rate | 0.0000 |
| top_3_hit_rate | 0.0000 |
| top_5_hit_rate | 0.0000 |
