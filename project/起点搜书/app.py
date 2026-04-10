from flask import Flask, request, render_template, jsonify
from query_processor import preprocess_query
from rag_retriever import retrieve_books
from llm_reason import generate_reason
import json
import os
from typing import List, Dict, Any

app = Flask(__name__, template_folder="templates", static_folder="static")

# 书籍数据文件路径
BOOKS_PATH = os.path.join(os.path.dirname(__file__), "data", "books.json")
# 加载书籍数据
with open(BOOKS_PATH, "r") as f:
    books: List[Dict[str, Any]] = json.load(f)


@app.route("/")
def index() -> str:
    """
    渲染首页，提供搜索界面。
    """
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend() -> Any:
    """
    处理前端POST请求，返回推荐书籍列表。
    请求体参数：
        query: str 查询关键词
        tags: List[str] 可选标签筛选
    返回：
        List[Dict] 推荐书籍信息
    """
    data: Dict = request.json
    query: str = data.get("query", "")
    tags: List[str] = data.get("tags", [])
    processed_query: str = preprocess_query(query)
    if not processed_query:
        return jsonify([])
    results: List[Dict[str, Any]] = retrieve_books(processed_query, books, tags)
    for book in results:
        book["reason"] = generate_reason(book, processed_query)
    return jsonify(results)


if __name__ == "__main__":
    # 启动Flask开发服务器
    app.run(debug=True)
