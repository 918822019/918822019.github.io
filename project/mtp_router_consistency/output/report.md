# MTP vs Router 一致性测试报告

**模型**: data/models/Ling-mini-base-2.0
**设备**: cuda
**max_new_tokens**: 8
**用时**: 54.7s
**测试样本数**: 2

## 聚合指标

| 指标 | 均值 | 最小值 | 最大值 |
|------|------|--------|--------|
| cosine_sim | 0.8584 | 0.8549 | 0.8618 |
| js_div | 0.5573 | 0.5379 | 0.5766 |
| kl_div | 4.8607 | 4.4863 | 5.2352 |
| spearman_rho | -0.0296 | -0.0435 | -0.0157 |
| top_1_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_3_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_5_hit_rate | 0.0000 | 0.0000 | 0.0000 |
| top_8_iou | 0.0111 | 0.0056 | 0.0167 |

## 各样本详情

### 样本 1

**Prompt**: `def fibonacci(n):`
**生成长度**: 8
**比较对数**: 12

| 指标 | 值 |
|------|-----|
| kl_div | 4.4863 |
| js_div | 0.5379 |
| cosine_sim | 0.8549 |
| top_8_iou | 0.0167 |
| spearman_rho | -0.0435 |
| top_1_hit_rate | 0.0000 |
| top_3_hit_rate | 0.0000 |
| top_5_hit_rate | 0.0000 |

**生成文本预览**:
```
def fibonacci(n):
    if n <= 0:
```

### 样本 2

**Prompt**: `def merge_sort(arr):`
**生成长度**: 8
**比较对数**: 12

| 指标 | 值 |
|------|-----|
| kl_div | 5.2352 |
| js_div | 0.5766 |
| cosine_sim | 0.8618 |
| top_8_iou | 0.0056 |
| spearman_rho | -0.0157 |
| top_1_hit_rate | 0.0000 |
| top_3_hit_rate | 0.0000 |
| top_5_hit_rate | 0.0000 |

**生成文本预览**:
```
def merge_sort(arr):
    if len(arr) <= 
```
