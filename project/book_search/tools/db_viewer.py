"""
数据库可视化管理界面
提供书籍和章节数据的浏览、搜索和统计功能

使用方法:
    1. 确保数据库文件存在: data/books.db
    2. 安装依赖: pip install flask
    3. 启动服务: python3 tools/db_viewer.py
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
from pathlib import Path

from src.process.polish_embedding_search import search_books_by_polish_embedding

# 项目根目录（tools/ 的上一级）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "books.db"))

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)


def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def table_exists(conn, table_name: str) -> bool:
    """检查指定数据表是否存在。"""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


@app.route("/")
def index():
    """主页 - 显示数据统计概览"""
    conn = get_db_connection()
    try:
        # 获取书籍统计
        book_stats = conn.execute(
            """
            SELECT 
                COUNT(*) as total_books,
                COALESCE(SUM(CASE WHEN chapter_count > 0 THEN 1 ELSE 0 END), 0) as catalog_ready_books,
                COALESCE(SUM(content_completed), 0) as content_completed_books
            FROM books
        """
        ).fetchone()

        # 获取章节统计
        chapter_stats = conn.execute(
            """
            SELECT 
                COUNT(*) as total_chapters,
                COALESCE(SUM(is_content_fetched), 0) as fetched_chapters
            FROM chapters
        """
        ).fetchone()

        polish_count = 0
        embed_count = 0
        if table_exists(conn, "book_polish"):
            polish_count = int(
                conn.execute("SELECT COUNT(*) AS c FROM book_polish").fetchone()["c"]
            )
        if table_exists(conn, "book_polish_embedding"):
            embed_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM book_polish_embedding"
                ).fetchone()["c"]
            )

        # 获取最近更新的书籍
        recent_books = conn.execute(
            """
            SELECT book_id, title, author, category, last_update, chapter_count
            FROM books
            WHERE last_update IS NOT NULL AND last_update != ''
            ORDER BY last_update DESC
            LIMIT 10
        """
        ).fetchall()

        stats = {
            "total_books": book_stats["total_books"],
            "catalog_ready": book_stats["catalog_ready_books"],
            "content_completed": book_stats["content_completed_books"],
            "total_chapters": chapter_stats["total_chapters"],
            "fetched_chapters": chapter_stats["fetched_chapters"],
            "pending_chapters": chapter_stats["total_chapters"]
            - chapter_stats["fetched_chapters"],
            "polished_books": polish_count,
            "embedded_books": embed_count,
            "recent_books": [dict(row) for row in recent_books],
        }

        return render_template("dashboard.html", stats=stats)
    finally:
        conn.close()


@app.route("/api/books")
def api_books():
    """API: 获取书籍列表(支持分页和搜索)"""
    conn = get_db_connection()
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        search = request.args.get("search", "", type=str)
        category = request.args.get("category", "", type=str)
        status = request.args.get("status", "", type=str)

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

        if status == "completed":
            conditions.append("content_completed = 1")
        elif status == "incomplete":
            conditions.append("content_completed = 0")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # 获取总数
        count_query = f"SELECT COUNT(*) as total FROM books WHERE {where_clause}"
        total = conn.execute(count_query, params).fetchone()["total"]

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

        return jsonify(
            {
                "success": True,
                "data": [dict(row) for row in books],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "pages": (total + per_page - 1) // per_page,
                },
            }
        )
    finally:
        conn.close()


@app.route("/api/books/<int:book_id>")
def api_book_detail(book_id):
    """API: 获取单本书的详细信息"""
    conn = get_db_connection()
    try:
        book = conn.execute(
            """
            SELECT * FROM books WHERE book_id = ?
        """,
            (book_id,),
        ).fetchone()

        if not book:
            return jsonify({"success": False, "error": "书籍不存在"}), 404

        # 获取章节列表
        chapters = conn.execute(
            """
            SELECT chapter_id, chapter_name, is_content_fetched, content_length
            FROM chapters
            WHERE book_id = ?
            ORDER BY chapter_id
        """,
            (book_id,),
        ).fetchall()

        return jsonify(
            {
                "success": True,
                "book": dict(book),
                "chapters": [dict(row) for row in chapters],
            }
        )
    finally:
        conn.close()


@app.route("/api/chapters/<int:book_id>/<int:chapter_id>")
def api_chapter_content(book_id, chapter_id):
    """API: 获取章节内容"""
    conn = get_db_connection()
    try:
        chapter = conn.execute(
            """
            SELECT * FROM chapters 
            WHERE book_id = ? AND chapter_id = ?
        """,
            (book_id, chapter_id),
        ).fetchone()

        if not chapter:
            return jsonify({"success": False, "error": "章节不存在"}), 404

        return jsonify({"success": True, "chapter": dict(chapter)})
    finally:
        conn.close()


@app.route("/api/categories")
def api_categories():
    """API: 获取所有分类"""
    conn = get_db_connection()
    try:
        categories = conn.execute(
            """
            SELECT DISTINCT category
            FROM books
            WHERE category IS NOT NULL AND category != ''
            ORDER BY category
        """
        ).fetchall()

        return jsonify(
            {"success": True, "categories": [row["category"] for row in categories]}
        )
    finally:
        conn.close()


@app.route("/api/statistics")
def api_statistics():
    """API: 获取详细统计数据"""
    conn = get_db_connection()
    try:
        # 按分类统计
        category_stats = conn.execute(
            """
            SELECT category, 
                   COUNT(*) as book_count,
                   SUM(chapter_count) as total_chapters,
                   SUM(CASE WHEN content_completed = 1 THEN 1 ELSE 0 END) as completed_books
            FROM books
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category
            ORDER BY book_count DESC
        """
        ).fetchall()

        # 按状态统计
        status_stats = conn.execute(
            """
            SELECT serial_status,
                   COUNT(*) as book_count
            FROM books
            WHERE serial_status IS NOT NULL AND serial_status != ''
            GROUP BY serial_status
            ORDER BY book_count DESC
        """
        ).fetchall()

        # 章节完成度分布
        completion_stats = conn.execute(
            """
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
        """
        ).fetchall()

        return jsonify(
            {
                "success": True,
                "category_stats": [dict(row) for row in category_stats],
                "status_stats": [dict(row) for row in status_stats],
                "completion_stats": [dict(row) for row in completion_stats],
            }
        )
    finally:
        conn.close()


@app.route("/api/polish-search", methods=["POST"])
def api_polish_search():
    """API: 基于润色 embedding 的相似书检索。"""
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    model_name = payload.get("model")
    top_k = int(payload.get("top_k", 10) or 10)

    if not query:
        return jsonify({"success": False, "error": "query 不能为空"}), 400

    try:
        results = search_books_by_polish_embedding(
            db_path=Path(DB_PATH),
            query=query,
            model_name=model_name,
            top_k=max(top_k, 1),
        )
        return jsonify({"success": True, "query": query, "results": results})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/processed-books")
def api_processed_books():
    """API: 获取处理后数据（润色 + embedding）列表。"""
    conn = get_db_connection()
    try:
        if not table_exists(conn, "book_polish"):
            return jsonify(
                {
                    "success": True,
                    "data": [],
                    "pagination": {
                        "page": 1,
                        "per_page": 20,
                        "total": 0,
                        "pages": 0,
                    },
                    "meta": {
                        "has_polish_table": False,
                        "has_embedding_table": table_exists(
                            conn, "book_polish_embedding"
                        ),
                    },
                }
            )

        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        search = request.args.get("search", "", type=str).strip()
        processed_filter = request.args.get("processed", "all", type=str)

        has_embedding_table = table_exists(conn, "book_polish_embedding")

        conditions = ["1=1"]
        params = []

        if search:
            conditions.append(
                "(b.title LIKE ? OR p.polished_title LIKE ? OR p.polished_intro LIKE ?)"
            )
            kw = f"%{search}%"
            params.extend([kw, kw, kw])

        if processed_filter == "embedded":
            if has_embedding_table:
                conditions.append("e.book_id IS NOT NULL")
            else:
                conditions.append("1=0")
        elif processed_filter == "polished":
            conditions.append("p.book_id IS NOT NULL")

        where_clause = " AND ".join(conditions)

        join_embedding = (
            "LEFT JOIN book_polish_embedding e ON e.book_id = b.book_id"
            if has_embedding_table
            else "LEFT JOIN (SELECT NULL AS book_id, NULL AS embedding_dim, NULL AS updated_at, NULL AS model_name) e ON 1=0"
        )

        count_sql = f"""
            SELECT COUNT(*) AS total
            FROM books b
            LEFT JOIN book_polish p ON p.book_id = b.book_id
            {join_embedding}
            WHERE {where_clause}
        """
        total = int(conn.execute(count_sql, params).fetchone()["total"])

        offset = (max(page, 1) - 1) * max(per_page, 1)
        data_sql = f"""
            SELECT
                b.book_id,
                b.title AS source_title,
                b.intro AS source_intro,
                b.author,
                b.category,
                b.homepage_url,
                p.polished_title,
                p.polished_intro,
                p.updated_at AS polish_updated_at,
                p.model_name AS polish_model,
                e.embedding_dim,
                e.updated_at AS embedding_updated_at,
                e.model_name AS embedding_model,
                CASE WHEN e.book_id IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM books b
            LEFT JOIN book_polish p ON p.book_id = b.book_id
            {join_embedding}
            WHERE {where_clause}
            ORDER BY b.book_id
            LIMIT ? OFFSET ?
        """
        query_params = params + [max(per_page, 1), offset]
        rows = conn.execute(data_sql, query_params).fetchall()

        return jsonify(
            {
                "success": True,
                "data": [dict(r) for r in rows],
                "pagination": {
                    "page": max(page, 1),
                    "per_page": max(per_page, 1),
                    "total": total,
                    "pages": (total + max(per_page, 1) - 1) // max(per_page, 1),
                },
                "meta": {
                    "has_polish_table": True,
                    "has_embedding_table": has_embedding_table,
                },
            }
        )
    finally:
        conn.close()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
