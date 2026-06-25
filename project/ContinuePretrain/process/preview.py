import sqlite3


def read_and_print_db(db_path: str):
    """读取SQLite数据库中的所有表及其数据并打印"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 获取所有用户表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        print("数据库中没有表。")
        conn.close()
        return

    # 2. 遍历每张表，打印表结构和数据
    for table in tables:
        print(f"\n{'='*60}")
        print(f"📋 表: {table}")
        print('='*60)

        # 打印列名
        cursor.execute(f"PRAGMA table_info('{table}');")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"列: {columns}")
        print('-'*60)

        # 打印所有行
        cursor.execute(f"SELECT * FROM '{table}';")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print(row)
        else:
            print("(空表)")

    conn.close()


if __name__ == "__main__":
    db_file = ".db"  # ← 修改为你的数据库文件路径
    read_and_print_db(db_file)