# SVD量化工具

基于奇异值分解(SVD)的模型压缩和量化工具，专门针对混合专家(MoE)模型架构设计。

## 目录结构

```
svd_quant/
├── src/                    # 核心源代码
│   ├── main.py            # 主要的SVD量化器实现
│   ├── compress_qwen_0_8b.py  # Qwen 0.8B压缩脚本
│   ├── inspect_model_keys.py  # 模型键名检查工具
│   └── __init__.py        # 包初始化文件
├── tests/                  # 测试文件
│   ├── core/              # 核心功能测试
│   ├── advanced/          # 高级量化方法测试
│   ├── analysis/          # 分析测试
│   ├── distillation/      # 蒸馏相关测试
│   ├── verification/      # 验证测试
│   └── misc/              # 其他测试
├── results/               # 结果和总结文件
│   ├── summaries/         # 总结文件
│   ├── analysis/          # 分析结果
│   └── reports/           # 测试报告
├── output/                # 输出文件（量化模型 .bin）
├── docs/                  # 文档
└── config/                # 配置文件占位（尚无实际配置文件）
```

## 功能特性

### 1. 模型低秩分解
- **非均匀秩分配策略**：针对不同投影层使用不同的分解秩
  - `gate_proj`层：默认秩32，实现更高压缩比
  - `up_proj`层：默认秩32，实现更高压缩比  
  - `down_proj`层：默认秩64，保持更好精度
- **智能层识别**：自动识别MoE专家层、注意力层、共享专家层
- **保持原始结构**：注意力层和共享专家不进行分解，直接量化

### 2. 混合精度量化
- **INT8量化**：用于SVD分解后的U_merged矩阵，精度更高
- **INT4量化**：用于SVD分解后的Vh矩阵，压缩比更高
- **对称量化**：使用对称量化方法，范围[-128,127]（INT8）和[-8,7]（INT4）
- **分组量化**：支持可配置的量化组大小（默认128）

### 3. 自定义二进制格式
- **高效存储**：支持INT8/INT4混合精度存储
- **完整元数据**：包含魔数、版本号、层数、层名、原始形状等
- **快速加载**：设计便于快速加载和推理
- **偏置项保存**：保持FP16精度

### 4. 量化质量评估
- **权重级别指标**：MSE、MAE、RMSE
- **模型级别指标**：困惑度（需要模型和验证数据）
- **压缩比计算**：原始大小与量化后大小的比率
- **误差监控**：相对误差阈值警告

## 快速开始

### 运行示例
```bash
# 运行主程序，查看示例输出
python src/main.py

# 运行Qwen 0.8B模型压缩
python src/compress_qwen_0_8b.py

# 运行测试
python -m pytest tests/core/
```

### 基本使用流程
1. 配置量化参数
2. 加载模型状态字典
3. 执行SVD分解和量化
4. 保存量化后的模型
5. 评估量化质量

## 测试文件说明

### 核心测试 (`tests/core/`)
- `test_simple.py` - 简单测试，基本SVD分解和量化功能
- `test_svd_quant.py` - 完整测试套件，覆盖所有主要功能
- `test_evaluation.py` - 评估测试，量化质量评估和困惑度计算
- `test_qwen.py` - Qwen模型测试

### 高级量化测试 (`tests/advanced/`)
- `test_fsq.py` - FSQ (Finite Scalar Quantization) 实现
- `test_fsq_v2.py` - FSQ 修正版本
- `test_logits_distillation.py` - Logits KL 蒸馏实现
- `test_hybrid_rvq.py` - 混合粒度 RVQ 实现

### 分析测试 (`tests/analysis/`)
- `test_moe_rank_analysis.py` - MoE秩分析测试
- `test_moe_rank_analysis_fixed.py` - MoE秩分析修复版本
- `test_layer_rank_analysis.py` - 层秩分析测试
- `test_singular_values.py` - 奇异值分析测试

### 蒸馏测试 (`tests/distillation/`)
- `test_expert_subnetwork_distillation.py` - 单Expert子网络蒸馏实现
- `test_logits_cache_distillation.py` - Logits缓存+离线蒸馏实现

### 验证测试 (`tests/verification/`)
- `test_verification_metrics.py` - 验证指标实现
- `test_verify_svd.py` - SVD验证测试
- `test_end_to_end_error.py` - 端到端误差测试

### 其他测试 (`tests/misc/`)
- 包含各种实验性和探索性测试文件

## 结果和分析

### 总结文件 (`results/summaries/`)
- `advanced_quantization_summary.txt` - 高级量化方法测试总结
- `fsql_distillation_implementation_summary.txt` - FSQ蒸馏实现总结
- `moe_rank_analysis_results.txt` - MoE秩分析结果

### 分析结果 (`results/analysis/`)
- `rvq_semantic_layering_analysis.png` - RVQ语义分层分析图
- `rvq_semantic_layering_conclusion.txt` - RVQ语义分层结论
- `rvq_semantic_layering_results.txt` - RVQ语义分层结果
- `verification_metrics_results.txt` - 验证指标结果

## 安装要求

### Python环境
- Python 3.8+
- PyTorch 1.9+
- NumPy 1.20+

### 依赖包
```bash
pip install torch numpy
```

### 可选依赖
- `transformers`：用于困惑度评估
- `tqdm`：用于进度显示

## 配置选项

### DecompositionConfig参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gate_proj_rank` | int | 32 | gate_proj层的分解秩 |
| `up_proj_rank` | int | 32 | up_proj层的分解秩 |
| `down_proj_rank` | int | 64 | down_proj层的分解秩 |
| `max_relative_error` | float | 1e-3 | 最大允许相对误差阈值 |
| `quant_group_size` | int | 128 | 量化组大小，必须是2的幂次 |
| `enable_qat` | bool | False | 是否启用量化感知微调（预留功能） |
| `qat_samples` | int | 1000 | 量化感知微调样本数量 |

### 秩选择建议
- **压缩优先**：使用较小的秩（16-32）
- **精度优先**：使用较大的秩（64-128）
- **平衡策略**：gate_proj和up_proj使用较小秩，down_proj使用较大秩

### 量化组大小选择
- **精度优先**：较小的组大小（32-64）
- **存储优先**：较大的组大小（128-256）
- **计算效率**：通常选择128或256

## 注意事项

### 适用场景
- 主要针对MoE模型设计，也可用于普通MLP层
- 适合需要模型压缩的推理场景
- 适合存储空间受限的部署环境

### 精度考虑
- 秩的选择需要平衡压缩比和精度
- 量化误差会随着组大小的增加而增加
- 建议通过实验确定最佳配置

### 兼容性
- 二进制格式需要配套的加载器
- 需要PyTorch和NumPy环境
- 支持FP32和FP16输入数据

### 限制
- 目前不支持动态秩调整
- 量化感知微调为预留功能
- 不支持模型架构自动识别

## 许可证

本工具为开源项目，遵循项目根目录的许可证。
