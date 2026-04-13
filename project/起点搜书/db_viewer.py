"""
数据库可视化管理界面
提供书籍和章节数据的浏览、搜索和统计功能

使用方法:
    1. 确保数据库文件存在: data/books.db
    2. 安装依赖: pip install flask
    3. 启动服务: python3 db_viewer.py
    4. 访问界面: http://localhost:5000

功能:
    - 数据统计面板: 展示书籍总数、章节数、完成进度等
    - 智能搜索: 支持按书名、作者、简介搜索
    - 分类筛选: 按书籍分类和完成状态过滤
    - 进度可视化: 显示每本书的章节抓取进度
    - 最近更新: 展示最近更新的书籍列表

API接口:
    GET /                    - 主页(统计面板)
    GET /api/books           - 获取书籍列表(支持分页和搜索)
    GET /api/books/<id>      - 获取单本书详情
    GET /api/chapters/<book_id>/<chapter_id> - 获取章节内容
    GET /api/categories      - 获取所有分类
    GET /api/statistics      - 获取详细统计数据
"""

import sqlite3
import os
from flask import Flask, render_template, request, jsonify
from datetime import datetime

app = Flask(__name__)

# 数据库路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'books.db')


def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    """主页 - 显示数据统计概览"""
    conn = get_db_connection()
    try:
        # 获取书籍统计
        book_stats = conn.execute("""
            SELECT 
                COUNT(*) as total_books,
                COALESCE(SUM(CASE WHEN chapter_count > 0 THEN 1 ELSE 0 END), 0) as catalog_ready_books,
                COALESCE(SUM(content_completed), 0) as content_completed_books
            FROM books
        """).fetchone()
        
        # 获取章节统计
        chapter_stats = conn.execute("""
            SELECT 
                COUNT(*) as total_chapters,
                COALESCE(SUM(is_content_fetched), 0) as fetched_chapters
            FROM chapters
        """).fetchone()
        
        # 获取最近更新的书籍
        recent_books = conn.execute("""
            SELECT book_id, title, author, category, last_update, chapter_count
            FROM books
            WHERE last_update IS NOT NULL AND last_update != ''
            ORDER BY last_update DESC
            LIMIT 10
        """).fetchall()
        
        stats = {
            'total_books': book_stats['total_books'],
            'catalog_ready': book_stats['catalog_ready_books'],
            'content_completed': book_stats['content_completed_books'],
            'total_chapters': chapter_stats['total_chapters'],
            'fetched_chapters': chapter_stats['fetched_chapters'],
            'pending_chapters': chapter_stats['total_chapters'] - chapter_stats['fetched_chapters'],
            'recent_books': [dict(row) for row in recent_books]
        }
        
        return render_template('dashboard.html', stats=stats)
    finally:
        conn.close()


@app.route('/api/books')
def api_books():
    """API: 获取书籍列表(支持分页和搜索)"""
    conn = get_db_connection()
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '', type=str)
        category = request.args.get('category', '', type=str)
        status = request.args.get('status', '', type=str)
        
        # 构建查询条件
        conditions = []
        params = []
        
        if search:
            conditions.append("(title LIKE ? OR author LIKE ? OR intro LIKE ?)")
            search_term = f"%{search}%"
            params.extend([search_term, search_term, search_term])
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if status == 'completed':
            conditions.append("content_completed = 1")
        elif status == 'incomplete':
            conditions.append("content_completed = 0")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # 获取总数
        count_query = f"SELECT COUNT(*) as total FROM books WHERE {where_clause}"
        total = conn.execute(count_query, params).fetchone()['total']
        
        # 获取分页数据
        offset = (page - 1) * per_page
        query = f"""
            SELECT book_id, title, author, category, serial_status, 
                   chapter_count, content_fetched_chapters, content_completed,
                   last_update, homepage_url
            FROM books
            WHERE {where_clause}
            ORDER BY book_id
            LIMIT ? OFFSET ?
        """
        params.extend([per_page, offset])
        
        books = conn.execute(query, params).fetchall()
        
        return jsonify({
            'success': True,
            'data': [dict(row) for row in books],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
    finally:
        conn.close()


@app.route('/api/books/<int:book_id>')
def api_book_detail(book_id):
    """API: 获取单本书的详细信息"""
    conn = get_db_connection()
    try:
        book = conn.execute("""
            SELECT * FROM books WHERE book_id = ?
        """, (book_id,)).fetchone()
        
        if not book:
            return jsonify({'success': False, 'error': '书籍不存在'}), 404
        
        # 获取章节列表
        chapters = conn.execute("""
            SELECT chapter_id, chapter_name, is_content_fetched, content_length
            FROM chapters
            WHERE book_id = ?
            ORDER BY chapter_id
        """, (book_id,)).fetchall()
        
        return jsonify({
            'success': True,
            'book': dict(book),
            'chapters': [dict(row) for row in chapters]
        })
    finally:
        conn.close()


@app.route('/api/chapters/<int:book_id>/<int:chapter_id>')
def api_chapter_content(book_id, chapter_id):
    """API: 获取章节内容"""
    conn = get_db_connection()
    try:
        chapter = conn.execute("""
            SELECT * FROM chapters 
            WHERE book_id = ? AND chapter_id = ?
        """, (book_id, chapter_id)).fetchone()
        
        if not chapter:
            return jsonify({'success': False, 'error': '章节不存在'}), 404
        
        return jsonify({
            'success': True,
            'chapter': dict(chapter)
        })
    finally:
        conn.close()


@app.route('/api/categories')
def api_categories():
    """API: 获取所有分类"""
    conn = get_db_connection()
    try:
        categories = conn.execute("""
            SELECT DISTINCT category
            FROM books
            WHERE category IS NOT NULL AND category != ''
            ORDER BY category
        """).fetchall()
        
        return jsonify({
            'success': True,
            'categories': [row['category'] for row in categories]
        })
    finally:
        conn.close()


@app.route('/api/statistics')
def api_statistics():
    """API: 获取详细统计数据"""
    conn = get_db_connection()
    try:
        # 按分类统计
        category_stats = conn.execute("""
            SELECT category, 
                   COUNT(*) as book_count,
                   SUM(chapter_count) as total_chapters,
                   SUM(CASE WHEN content_completed = 1 THEN 1 ELSE 0 END) as completed_books
            FROM books
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category
            ORDER BY book_count DESC
        """).fetchall()
        
        # 按状态统计
        status_stats = conn.execute("""
            SELECT serial_status,
                   COUNT(*) as book_count
            FROM books
            WHERE serial_status IS NOT NULL AND serial_status != ''
            GROUP BY serial_status
            ORDER BY book_count DESC
        """).fetchall()
        
        # 章节完成度分布
        completion_stats = conn.execute("""
            SELECT 
                CASE 
                    WHEN chapter_count = 0 THEN '0%'
                    WHEN CAST(content_fetched_chapters AS FLOAT) / chapter_count < 0.25 THEN '0-25%'
                    WHEN CAST(content_fetched_chapters AS FLOAT) / chapter_count < 0.5 THEN '25-50%'
                    WHEN CAST(content_fetched_chapters AS FLOAT) / chapter_count < 0.75 THEN '50-75%'
                    WHEN CAST(content_fetched_chapters AS FLOAT) / chapter_count < 1 THEN '75-99%'
                    ELSE '100%'
                END as completion_range,
                COUNT(*) as book_count
            FROM books
            WHERE chapter_count > 0
            GROUP BY completion_range
            ORDER BY completion_range
        """).fetchall()
        
        return jsonify({
            'success': True,
            'category_stats': [dict(row) for row in category_stats],
            'status_stats': [dict(row) for row in status_stats],
            'completion_stats': [dict(row) for row in completion_stats]
        })
    finally:
        conn.close()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
