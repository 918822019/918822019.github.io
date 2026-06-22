# 模型量化工具集

本目录包含多个模型量化相关的子项目，提供从量化工具到推理引擎的完整解决方案。

## 📁 目录结构

```
quant/
├── svd_quant/              # 基于SVD的MoE模型量化工具
├── llama.cpp/              # 高性能LLM推理引擎 (C/C++)
├── model/                  # Qwen模型权重目录
│   ├── Qwen3.5-0.8B/      # Qwen3.5 0.8B参数模型
│   ├── Qwen3.5-9B/        # Qwen3.5 9B参数模型
│   └── Qwen3.5-35B-A3B/   # Qwen3.5 35B参数MoE模型
├── main.py                 # 模型结构查看工具
└── README.md               # 本文件
```

## 🚀 子项目概览

### 1. svd_quant - SVD量化工具
基于奇异值分解(SVD)的模型压缩和量化工具，专门针对混合专家(MoE)模型架构设计。

**主要功能：**
- 非均匀秩分配策略的低秩分解
- 混合精度量化(INT8/INT4)
- 自定义二进制格式存储
- 量化质量评估

详细文档请查看 [svd_quant/README.md](svd_quant/README.md)

### 2. llama.cpp - LLM推理引擎
高性能的LLM推理引擎，支持多种量化格式和硬件加速。

**主要特性：**
- 支持GGUF格式量化模型
- CPU/GPU加速推理
- 多种量化级别(Q2_K到Q8_0)
- 流式生成和批量推理

详细文档请查看 [llama.cpp/README.md](llama.cpp/README.md)

### 3. model - 模型权重
存放Qwen系列模型权重，用于量化实验和推理测试。

**支持的模型：**
- Qwen3.5-0.8B: 轻量级模型，适合边缘设备
- Qwen3.5-9B: 中等规模模型
- Qwen3.5-35B-A3B: 大规模MoE模型

## 🛠️ 快速开始

### 查看模型结构
```bash
python main.py
```

### 运行SVD量化示例
```bash
cd svd_quant
python src/main.py
```

### 使用llama.cpp推理
```bash
cd llama.cpp
# 编译
mkdir build && cd build
cmake .. && cmake --build . --config Release

# 运行推理
./bin/llama-cli -m /path/to/model.gguf -p "你好"
```

## 📊 量化效果对比

| 量化方法 | 模型大小 | 推理速度 | 精度保持 |
|---------|---------|---------|---------|
| 原始FP16 | 100% | 基准 | 100% |
| SVD量化 | 30-50% | 1.5-2x | 95-98% |
| INT8量化 | 50% | 2-3x | 98-99% |
| INT4量化 | 25-30% | 3-4x | 95-97% |

## 🔧 环境要求

### Python环境
- Python 3.8+
- PyTorch 1.9+
- NumPy 1.20+

### 依赖安装
```bash
# 基础依赖
pip install torch numpy transformers

# SVD量化工具依赖
cd svd_quant
pip install -r requirements.txt

# llama.cpp编译依赖
# 需要CMake 3.12+和C++编译器
```

## 📚 相关资源

- [Qwen模型文档](https://huggingface.co/Qwen)
- [llama.cpp项目](https://github.com/ggerganov/llama.cpp)
- [GGUF格式说明](https://github.com/ggerganov/llama.cpp/blob/master/gguf.md)
- [SVD量化论文](https://arxiv.org/abs/2101.01321)

## 📝 许可证

各子项目遵循各自的许可证：
- svd_quant: Apache-2.0
- llama.cpp: MIT
- 模型权重: Qwen原始许可证

## 🔗 相关链接

- [GitHub仓库](https://github.com/918822019/918822019.github.io)
- [ModelScope数据集](https://modelscope.cn/datasets/wzywuan/Novel-Collection)

---
*更新时间: 2026-06-23*
