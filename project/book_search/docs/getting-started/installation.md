# 安装指南

## 系统要求

- **Python**: 3.11 或更高版本
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议 4GB 以上（处理大量书籍时需要更多）
- **磁盘**: 根据数据量预留空间（每本书约 50KB-200KB）

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/918822019.github.io.git
cd 918822019.github.io/project/book_search
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. 安装依赖

#### 运行依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含以下核心依赖：

| 包名 | 用途 |
|------|------|
| `modelscope` | ModelScope 数据集上传/下载 |
| `numpy` | 向量计算 |
| `faiss-cpu` | 向量索引（需单独安装） |
| `PyYAML` | 配置文件解析 |

#### 开发依赖

```bash
pip install -r requirements-dev.txt
```

| 包名 | 用途 |
|------|------|
| `pytest` | 单元测试 |
| `pytest-cov` | 测试覆盖率 |
| `black` | 代码格式化 |
| `flake8` | 代码检查 |
| `isort` | import 排序 |

### 4. 安装 Faiss

Faiss 需要单独安装，推荐使用 CPU 版本：

```bash
# 方式一：通过 pip 安装
pip install faiss-cpu

# 方式二：如果需要 GPU 支持
# pip install faiss-gpu
```

!!! note "Faiss 安装说明"
    `faiss-cpu` 不在 `requirements.txt` 中，需要手动安装。如果安装失败，可以尝试：
    ```bash
    pip install faiss-cpu --no-cache-dir
    ```
    或使用 conda：
    ```bash
    conda install -c conda-forge faiss-cpu
    ```

### 5. 配置环境变量

复制示例配置文件并填写你的 API 密钥：

```bash
# 创建 .env 文件
cat > .env << 'EOF'
# LLM 配置
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

# Embedding 配置
EMBEDDING_API_KEY=your-api-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_NAME=text-embedding-3-small

# Reranker 配置（可选，不配置则使用默认）
RERANKER_API_KEY=your-api-key
RERANKER_BASE_URL=https://api.openai.com/v1
RERANKER_MODEL_NAME=text-embedding-3-small

# 请求超时（秒）
REQUEST_TIMEOUT=60
EOF
```

详细的环境变量说明请参考 [环境变量配置](../configuration/environment.md)。

## 验证安装

运行以下命令验证安装是否成功：

```bash
# 检查 Python 版本
python --version  # 应 >= 3.11

# 检查核心依赖
python -c "import faiss; print(f'Faiss 版本: {faiss.__version__}')"
python -c "import numpy; print(f'NumPy 版本: {numpy.__version__}')"

# 运行全部测试（69 个用例，68 通过，1 跳过 API 在线测试）
pytest tests/ -v
```

## 常见问题

### Faiss 安装失败

```bash
# 尝试指定版本
pip install faiss-cpu==1.7.4

# 或使用 conda
conda install -c conda-forge faiss-cpu
```

### 权限问题

```bash
# 使用用户级安装
pip install --user -r requirements.txt
```

### 代理环境

```bash
# 使用代理
pip install -r requirements.txt --proxy http://proxy-host:port
```

## 下一步

安装完成后，参考 [快速上手](quickstart.md) 开始使用系统。
