#!/bin/bash
# ============================================================================
# 数据库可视化界面启动脚本
# ============================================================================
#
# 使用方法:
#     chmod +x start_viewer.sh
#     ./start_viewer.sh
#
# 功能:
#     - 自动检查Python环境
#     - 自动安装Flask依赖
#     - 验证数据库文件存在性
#     - 启动Web服务 (默认端口: 5000)
#
# 访问地址:
#     http://localhost:5000
#
# 注意事项:
#     - 确保数据库文件 data/books.db 已存在
#     - 如果数据库不存在，先运行数据抓取脚本
#     - 按 Ctrl+C 停止服务器
#
# ============================================================================

echo "======================================"
echo "  起点搜书 - 数据库可视化管理界面"
echo "======================================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装 Flask..."
    pip3 install flask
fi

# 检查数据库文件
DB_PATH="data/books.db"
if [ ! -f "$DB_PATH" ]; then
    echo "警告: 数据库文件 $DB_PATH 不存在"
    echo "请先运行数据抓取脚本生成数据库"
    echo ""
    echo "示例命令:"
    echo "  cd data_get"
    echo "  python main.py crawl-books --start 1 --end 100"
    echo ""
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 启动服务
echo ""
echo "启动服务器..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo ""

python3 tools/db_viewer.py
