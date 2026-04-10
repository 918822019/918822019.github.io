def preprocess_query(query):
    # 简单过滤和改写逻辑
    if query in ["你好", "在吗", "hi", "hello"]:
        return None
    return query.strip()
