"""
Agent 服务完整业务流程
检查数据是否已向量化，如未向量化则执行向量化，然后进入问答阶段
"""
import sqlite3
from pathlib import Path
from typing import List

import numpy as np

from src.agent import Agent
from src.config import config
from src.process.polish import (
    EMBED_TABLE,
    POLISH_TABLE,
    ensure_embedding_table,
    ensure_polish_table,
    get_faiss_index_path,
    load_or_create_faiss_index,
    resolve_existing_db_path,
)
from src.process.polish._faiss import _require_faiss


class BookSearchEngine:
    """小说搜索引引擎 - 整合向量化和检索功能"""

    def __init__(self, db_path: str = "../../data/book_search/books.db"):
        self.db_path = Path(db_path)
        self.agent = Agent()
        self.faiss_index = None
        self.conn = None
        
    def connect(self):
        """连接数据库并初始化"""
        active_db_path = resolve_existing_db_path(self.db_path)
        if not active_db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {active_db_path}")
        
        self.conn = sqlite3.connect(str(active_db_path))
        self.conn.row_factory = sqlite3.Row
        
        # 确保表结构存在
        ensure_polish_table(self.conn)
        ensure_embedding_table(self.conn)
        
        print(f"✅ 数据库连接成功: {active_db_path}")
        
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            
    def check_vectorization_status(self) -> dict:
        """检查数据向量化状态"""
        if not self.conn:
            self.connect()
            
        # 统计润色后的书籍总数
        polish_count = self.conn.execute(
            f"SELECT COUNT(*) as cnt FROM {POLISH_TABLE}"
        ).fetchone()["cnt"]
        
        # 统计已向量化的书籍数
        embed_count = self.conn.execute(
            f"SELECT COUNT(*) as cnt FROM {EMBED_TABLE}"
        ).fetchone()["cnt"]
        
        status = {
            "polished_books": polish_count,
            "embedded_books": embed_count,
            "need_embedding": polish_count - embed_count,
            "is_fully_embedded": polish_count > 0 and polish_count == embed_count
        }
        
        return status
        
    def vectorize_books(self, limit: int = 0, overwrite: bool = False) -> dict:
        """
        对未向量化的书籍执行向量化
        
        Args:
            limit: 最多处理多少本，0 表示不限制
            overwrite: 是否覆盖已有的 embedding
            
        Returns:
            处理统计信息
        """
        if not self.conn:
            self.connect()
            
        from src.process.polish import run_polish_embedding
        
        print("\n🔄 开始向量化处理...")
        stats = run_polish_embedding(
            db_path=self.db_path,
            model_name=None,
            limit=limit,
            sleep_seconds=0.1,  # 避免 API 限流
            overwrite=overwrite
        )
        
        print(f"\n✅ 向量化完成:")
        print(f"   总记录数: {stats['total']}")
        print(f"   本次处理: {stats['processed']}")
        print(f"   成功写入: {stats['changed']}")
        print(f"   跳过数量: {stats['skipped']}")
        print(f"   失败数量: {stats['failed']}")
        
        return stats
        
    def load_faiss_index(self):
        """加载 Faiss 向量索引"""
        if not self.conn:
            self.connect()
            
        active_db_path = resolve_existing_db_path(self.db_path)
        index_path = get_faiss_index_path(active_db_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Faiss 索引文件不存在: {index_path}")
        
        # 从数据库获取 embedding 维度
        dim_row = self.conn.execute(
            f"SELECT embedding_dim FROM {EMBED_TABLE} LIMIT 1"
        ).fetchone()
        
        if not dim_row:
            raise ValueError("数据库中没有 embedding 记录")
        
        dim = dim_row["embedding_dim"]
        self.faiss_index = load_or_create_faiss_index(index_path, dim=dim)
        
        print(f"✅ Faiss 索引加载成功: {index_path}")
        print(f"   索引维度: {dim}")
        print(f"   向量数量: {self.faiss_index.ntotal}")
        
    def search_books_by_query(self, query: str, top_k: int = 5) -> List[dict]:
        """
        根据查询搜索相关书籍
        
        Args:
            query: 用户查询
            top_k: 返回前 k 个结果
            
        Returns:
            相关书籍列表（包含元数据）
        """
        if not self.faiss_index:
            self.load_faiss_index()
            
        # 1. 将查询转换为向量并归一化
        query_embedding = np.asarray(self.agent.embed_text(query), dtype=np.float32).reshape(1, -1)
        faiss = _require_faiss()
        faiss.normalize_L2(query_embedding)
        
        # 2. 在 Faiss 中搜索相似向量
        ntotal = self.faiss_index.ntotal
        search_k = min(top_k * 2, ntotal) if ntotal > 0 else 0
        if search_k == 0:
            return []
        scores, indices = self.faiss_index.search(query_embedding, search_k)
        
        # 3. 批量从数据库获取书籍元数据
        candidate_ids = [int(bid) for bid in indices[0].tolist() if int(bid) >= 0]
        if not candidate_ids:
            return []
        
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self.conn.execute(
            f"""
            SELECT p.book_id, p.source_title, p.source_intro,
                   p.polished_title, p.polished_intro,
                   e.text_content
            FROM {POLISH_TABLE} p
            LEFT JOIN {EMBED_TABLE} e ON p.book_id = e.book_id
            WHERE p.book_id IN ({placeholders})
            """,
            candidate_ids
        ).fetchall()
        
        metadata_map = {int(r["book_id"]): r for r in rows}
        
        results = []
        for i, book_id in enumerate(candidate_ids):
            row = metadata_map.get(book_id)
            if row is None:
                continue
            results.append({
                "book_id": book_id,
                "score": float(scores[0][i]),
                "original_title": row["source_title"],
                "original_intro": row["source_intro"],
                "polished_title": row["polished_title"],
                "polished_intro": row["polished_intro"],
                "text_content": row["text_content"]
            })
        
        # 4. 使用 Reranker 精排
        if results:
            documents = [r["text_content"] for r in results]
            reranked = self.agent.rerank_documents(query, documents, top_k=top_k)
            
            final_results = []
            for rank_idx, (orig_idx, rerank_score) in enumerate(reranked):
                if orig_idx < len(results):
                    result = results[orig_idx].copy()
                    result["rerank_score"] = rerank_score
                    result["final_rank"] = rank_idx + 1
                    final_results.append(result)
            
            return final_results[:top_k]
        
        return results[:top_k]
        
    def answer_with_context(self, query: str, books: List[dict]) -> str:
        """
        基于搜索结果生成回答
        
        Args:
            query: 用户原始查询
            books: 相关书籍列表
            
        Returns:
            AI 生成的回答
        """
        if not books:
            return self.agent.process_query(query)
        
        # 构建上下文
        context_parts = []
        for i, book in enumerate(books, 1):
            context_parts.append(
                f"书籍{i}: 《{book['polished_title']}》\n"
                f"简介: {book['polished_intro']}\n"
                f"相关性分数: {book.get('rerank_score', book.get('score', 0)):.3f}"
            )
        
        context = "\n\n".join(context_parts)
        
        # 生成回答
        prompt = f"""基于以下推荐的书籍信息，回答用户的问题。

相关书籍：
{context}

用户问题：{query}

请根据上述书籍信息给出推荐和建议："""
        
        return self.agent.llm_client.generate(prompt)


def main():
    """主流程：检查向量化状态 -> 必要时向量化 -> 进入问答"""
    print("\n" + "=" * 70)
    print("📚 小说智能搜索系统")
    print("=" * 70 + "\n")
    
    engine = BookSearchEngine(db_path="../../data/book_search/books.db")
    
    try:
        # Step 1: 连接数据库
        print("Step 1: 连接数据库...")
        engine.connect()
        
        # Step 2: 检查向量化状态
        print("\nStep 2: 检查数据向量化状态...")
        status = engine.check_vectorization_status()
        
        print(f"   润色后书籍总数: {status['polished_books']}")
        print(f"   已向量化的书: {status['embedded_books']}")
        print(f"   待向量化数量: {status['need_embedding']}")
        
        # Step 3: 如果未完全向量化，执行向量化
        if not status['is_fully_embedded']:
            print("\n⚠️  检测到有书籍未向量化，开始执行向量化...")
            
            user_input = input("是否现在开始向量化？(y/n): ").strip().lower()
            if user_input == 'y':
                limit_input = input("处理数量限制（0=全部）: ").strip()
                limit = int(limit_input) if limit_input.isdigit() else 0
                
                engine.vectorize_books(limit=limit)
                
                # 重新检查状态
                status = engine.check_vectorization_status()
                print(f"\n✅ 当前向量化状态: {status['embedded_books']}/{status['polished_books']}")
            else:
                print("⏭️  跳过向量化步骤")
        else:
            print("\n✅ 所有书籍已完成向量化")
        
        # Step 4: 加载 Faiss 索引
        print("\nStep 3: 加载向量索引...")
        try:
            engine.load_faiss_index()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("提示: 请先执行向量化步骤")
            return
        
        # Step 5: 进入问答循环
        print("\n" + "=" * 70)
        print("💬 进入问答模式（输入 'quit' 或 'exit' 退出）")
        print("=" * 70 + "\n")
        
        while True:
            try:
                query = input("🔍 请输入您的问题: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 再见！")
                    break
                
                print("\n⏳ 正在搜索相关书籍...\n")
                
                # 搜索书籍
                books = engine.search_books_by_query(query, top_k=5)
                
                if not books:
                    print("❌ 未找到相关书籍\n")
                    continue
                
                # 显示搜索结果
                print(f"📖 找到 {len(books)} 本相关书籍:\n")
                for i, book in enumerate(books, 1):
                    print(f"{i}. 《{book['polished_title']}》")
                    print(f"   简介: {book['polished_intro'][:100]}...")
                    print(f"   相关性: {book.get('rerank_score', book.get('score', 0)):.3f}\n")
                
                # 生成回答
                print("⏳ 正在生成回答...\n")
                answer = engine.answer_with_context(query, books)
                
                print("💡 AI 推荐:")
                print("-" * 70)
                print(answer)
                print("-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()
                print()
    
    finally:
        engine.close()


if __name__ == "__main__":
    main()
