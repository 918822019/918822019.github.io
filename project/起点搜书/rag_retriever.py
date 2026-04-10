def retrieve_books(query, books, top_k=5):
    # 简单关键词匹配召回
    results = []
    for book in books:
        if query in book.get("tags", []) or query in book.get("name", ""):
            results.append(
                {
                    "name": book["name"],
                    "author": book["author"],
                    "description": book["description"],
                    "tags": book["tags"],
                    "link": book["link"],
                    "reason": f"推荐理由：标签包含 {query}",
                }
            )
        if len(results) >= top_k:
            break
    return results
