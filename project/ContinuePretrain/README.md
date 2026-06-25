# ContinuePretrain - 继续预训练项目

基于 Qwen3.5-0.8B 模型的继续预训练实验框架，包含数据清洗、去重、训练管道等核心组件。

## 📁 目录结构

```
ContinuePretrain/
├── main.py                     # 入口脚本
├── data/
│   └── books.db                # SQLite 数据库（小说数据）
├── Qwen3.5-0.8B/               # 基础模型权重
│   ├── config.json             # 模型配置
│   ├── model.safetensors       # 模型权重
│   ├── tokenizer.json          # 分词器
│   └── ...                     # 其他配置文件
└── process/                    # 数据处理管道
    ├── pipeline.py             # 核心：文本去噪去重处理器
    └── preview.py              # 数据库预览工具
```

## 🔧 核心功能

### 1. TextDenoiserDeduplicator - 文本去噪去重处理器 (`process/pipeline.py`)

专为网络小说、网页文本设计的清洗工具，支持两大核心功能：

#### 文本去噪
- **HTML/XML 标签移除** - 清理网页爬取残留的标签
- **控制字符清理** - 移除 Unicode 控制字符（C0/C1 范围）
- **章节标题移除** - 支持中英文多种格式（"第1章"、"Chapter 1" 等）
- **作者求票话术过滤** - 覆盖主流小说平台常见营销话术
- **无意义行过滤** - 移除纯数字行、过短行、空行
- **空行合并** - 优化段落结构

#### 段落级去重
- **MinHash 算法** - 使用 `datasketch` 库实现高效相似度检测
- **Jaccard 相似度** - 基于 n-gram 的语义相似度计算
- **可配置阈值** - 默认 0.85，支持动态调整
- **首次保留策略** - 保留首次出现的段落，移除后续重复

### 2. 数据库预览工具 (`process/preview.py`)
快速查看 SQLite 数据库表结构和数据内容。

## ⚡ 快速开始

### 安装依赖
```bash
pip install datasketch numpy
```

### 基础使用
```python
from process.pipeline import TextDenoiserDeduplicator

# 创建处理器（使用默认配置）
processor = TextDenoiserDeduplicator(
    minhash_threshold=0.85,    # 相似度阈值
    num_perm=128,              # MinHash 签名长度
    ngram_size=3,              # n-gram 大小
    min_paragraph_length=5     # 最小段落长度
)

# 完整处理流程：去噪 + 去重
result = processor.process(raw_text, verbose=True)

# 获取结果
cleaned_text = result['cleaned_text']    # 去噪后文本
deduped_text = result['deduped_text']    # 去重后最终文本
duplicates = result['duplicates_info']   # 重复段落信息
stats = result['stats']                  # 统计信息
```

### 单独使用去噪/去重
```python
# 仅去噪
cleaned = processor.clean_text(text, verbose=True)

# 仅去重（需先去噪）
deduped, dup_info = processor.paragraph_level_dedup(cleaned_text, verbose=True)
```

### 动态调整参数
```python
# 调整去重严格程度
processor.set_threshold(0.9)   # 更严格
processor.set_threshold(0.7)   # 更宽松

# 调整 n-gram 粒度
processor.set_ngram_size(2)    # 更粗粒度
processor.set_ngram_size(4)    # 更细粒度

# 查看当前配置
print(processor.config)
```

### 数据库预览
```bash
python process/preview.py
# 修改 preview.py 中的 db_file 路径
```

## ⚙️ 配置参数详解

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `minhash_threshold` | 0.85 | MinHash 相似度阈值，越大越严格 | 0.7 ~ 0.9 |
| `num_perm` | 128 | MinHash 排列数（哈希函数数量） | 64 ~ 256 |
| `ngram_size` | 3 | 文本分词 n-gram 大小 | 中文 2-3，英文 3-5 |
| `min_paragraph_length` | 5 | 最小段落长度阈值 | 3 ~ 10 |

## 📊 处理流程图

```
原始文本
    │
    ▼
┌─────────────────────────────────────┐
│         文本去噪                     │
├─────────────────────────────────────┤
│ 1. 移除 HTML/XML 标签               │
│ 2. 清理控制字符/乱码                │
│ 3. 移除章节标题                     │
│ 4. 过滤作者求票话术                 │
│ 5. 过滤无意义行（纯数字/过短）       │
│ 6. 合并连续空行                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         段落级去重                   │
├─────────────────────────────────────┤
│ 1. 按段落分割（双换行符）           │
│ 2. 生成 MinHash 签名                │
│ 3. 计算 Jaccard 相似度              │
│ 4. 阈值判定重复                     │
│ 5. 保留首次，移除后续               │
│ 6. 重新组合文本                     │
└─────────────────────────────────────┘
    │
    ▼
清洗后的高质量文本
```

## 📈 统计输出示例

```
处理完成 - 统计摘要
======================================================================
指标                  原始        去噪后       去重后
文本长度(字符)        15,234      12,100       9,876
段落数                -           45           32
----------------------------------------------------------------------
去噪减少: 20.6%
去重减少: 18.4%
重复段落: 13 个 (28.9%)
======================================================================
```

## 🎯 适用场景

- **网络小说数据清洗** - 起点、晋江等平台爬取数据
- **网页内容提取后处理** - 去除导航、广告、版权信息
- **大规模文本数据集预处理** - 训练数据去重、质量提升
- **避免模型背诵重复内容** - 防止"名场面"过拟合

## 🔗 相关链接

- [Qwen3.5-0.8B 模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [datasketch 文档](https://github.com/ekzhu/datasketch)
- [MinHash 算法原理](https://en.wikipedia.org/wiki/MinHash)

## 📝 许可证

本项目遵循 Apache-2.0 许可证，模型权重遵循 Qwen 原始许可证。