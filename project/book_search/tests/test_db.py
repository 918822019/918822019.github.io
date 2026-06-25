"""
数据库连接测试脚本
用于验证数据库文件是否存在且可访问

使用方法:
    cd tests
    python3 test_db.py

功能:
    - 检查数据库文件是否存在
    - 验证数据库连接
    - 显示数据统计概览
    - 展示示例数据
    - 检查数据库索引

输出:
    - ✅ 成功: 显示数据库信息和启动命令
    - ❌ 失败: 显示错误信息和解决建议
"""

import sqlite3
import os
import sys

# 数据库路径(相对于项目根目录)
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'books.db')
DB_PATH = os.path.normpath(DB_PATH)


def test_database():
    """测试数据库连接和基本查询"""
    
    print("=" * 60)
    print("数据库连接测试")
    print("=" * 60)
    print()
    
    # 检查文件是否存在
    if not os.path.exists(DB_PATH):
        print(f"❌ 错误: 数据库文件不存在: {DB_PATH}")
        print()
        print("请先运行数据抓取脚本:")
        print("  python -m src.crawler.engine crawl-books --start 1 --end 100")
        return False
    
    print(f"✅ 数据库文件存在: {DB_PATH}")
    print(f"   文件大小: {os.path.getsize(DB_PATH) / (1024*1024):.2f} MB")
    print()
    
    try:
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        print("✅ 数据库连接成功")
        print()
        
        # 检查表结构
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        
        print(f"📋 数据库中的表:")
        for table in tables:
            print(f"   - {table['name']}")
        print()
        
        # 统计信息
        book_count = conn.execute("SELECT COUNT(*) as count FROM books").fetchone()['count']
        chapter_count = conn.execute("SELECT COUNT(*) as count FROM chapters").fetchone()['count']
        
        print(f"📊 数据统计:")
        print(f"   - 书籍数量: {book_count:,}")
        print(f"   - 章节数量: {chapter_count:,}")
        print()
        
        # 显示示例数据
        if book_count > 0:
            sample_books = conn.execute("""
                SELECT book_id, title, author, category, chapter_count
                FROM books
                LIMIT 5
            """).fetchall()
            
            print(f"📚 示例书籍 (前5本):")
            for book in sample_books:
                print(f"   [{book['book_id']}] {book['title']}")
                print(f"       作者: {book['author'] or '未知'} | "
                      f"分类: {book['category'] or '未知'} | "
                      f"章节: {book['chapter_count']}")
            print()
        
        # 检查索引
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        
        if indexes:
            print(f"🔍 数据库索引:")
            for idx in indexes:
                print(f"   - {idx['name']}")
            print()
        
        conn.close()
        
        print("=" * 60)
        print("✅ 数据库测试通过!")
        print("=" * 60)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_database()
    sys.exit(0 if success else 1)
