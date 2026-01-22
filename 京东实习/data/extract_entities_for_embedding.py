"""
从数据文件中抽取实体名称字段的值，构建用于embedding的数据集

抽取的字段:
1. 实体信息表.csv - entity_chinese_name, entity_english_name, entity_alias
2. 基金业务主体数据.csv - fund_sht_name, fund_alias_name
3. 指标信息表_old.csv - indicator_chinese_name, indicator_english_name, indicator_alias
4. graph.json - display_name, name, name_en, abbr_name

输出: embedding_dataset.csv, embedding_dataset.json, embedding_texts.txt
"""

import pandas as pd
import json
import os
from typing import List, Dict, Set

# 获取当前脚本所在目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_column_values(file_path: str, columns: List[str], source_name: str) -> List[Dict]:
    """从CSV文件抽取指定列的值"""
    entities = []

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        for col in columns:
            if col in df.columns:
                values = df[col].dropna().unique()
                for val in values:
                    if isinstance(val, str) and val.strip():
                        entities.append({
                            'text': val.strip(),
                            'source': source_name,
                            'field': col
                        })

        print(f"从 {source_name} 抽取了 {len(entities)} 条记录")
    except Exception as e:
        print(f"读取 {source_name} 失败: {e}")

    return entities


def extract_from_entity_info() -> List[Dict]:
    """从实体信息表抽取实体名称"""
    file_path = os.path.join(DATA_DIR, "实体信息表.csv")
    columns = ['entity_chinese_name', 'entity_english_name', 'entity_alias']
    return extract_column_values(file_path, columns, "实体信息表")


def extract_from_fund_data() -> List[Dict]:
    """从基金业务主体数据抽取基金名称"""
    file_path = os.path.join(DATA_DIR, "基金业务主体数据.csv")
    columns = ['fund_sht_name', 'fund_alias_name']
    return extract_column_values(file_path, columns, "基金业务主体数据")


def extract_from_indicator_info() -> List[Dict]:
    """从指标信息表抽取指标名称"""
    file_path = os.path.join(DATA_DIR, "指标信息表_old.csv")
    columns = ['indicator_chinese_name', 'indicator_english_name', 'indicator_alias']
    return extract_column_values(file_path, columns, "指标信息表")


def extract_from_graph() -> List[Dict]:
    """从graph.json抽取实体名称"""
    file_path = os.path.join(DATA_DIR, "graph.json")
    entities = []
    fields = ['display_name', 'name', 'name_en', 'abbr_name']

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        seen_values: Dict[str, Set[str]] = {f: set() for f in fields}

        for item in graph_data:
            if isinstance(item, dict):
                for field in fields:
                    val = item.get(field, '')
                    if val and isinstance(val, str) and val.strip() and val not in seen_values[field]:
                        seen_values[field].add(val)
                        entities.append({
                            'text': val.strip(),
                            'source': 'graph.json',
                            'field': field
                        })

        print(f"从 graph.json 抽取了 {len(entities)} 条记录")
    except Exception as e:
        print(f"读取 graph.json 失败: {e}")

    return entities


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """对实体列表进行去重（基于text）"""
    seen_texts: Set[str] = set()
    unique_entities = []

    for entity in entities:
        text = entity['text']
        if text not in seen_texts:
            seen_texts.add(text)
            unique_entities.append(entity)

    return unique_entities


def save_dataset(entities: List[Dict], deduplicated: bool = True):
    """保存数据集为CSV和JSON格式"""
    if deduplicated:
        entities = deduplicate_entities(entities)
        print(f"\n去重后共有 {len(entities)} 条唯一实体名称")

    # 保存为CSV
    csv_path = os.path.join(DATA_DIR, "embedding_dataset.csv")
    df = pd.DataFrame(entities)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"数据集已保存到: {csv_path}")

    # 保存为JSON
    json_path = os.path.join(DATA_DIR, "embedding_dataset.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    print(f"数据集已保存到: {json_path}")

    # 保存纯文本列表 (只包含text字段，用于快速embedding)
    txt_path = os.path.join(DATA_DIR, "embedding_texts.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(entity['text'] + '\n')
    print(f"纯文本列表已保存到: {txt_path}")

    # 打印统计信息
    print("\n=== 数据集统计 ===")
    df_stats = pd.DataFrame(entities)
    print("\n按来源统计:")
    print(df_stats['source'].value_counts())
    print("\n按字段统计:")
    print(df_stats['field'].value_counts())
    print("\n前20条样例:")
    for i, entity in enumerate(entities[:20]):
        print(f"  {i+1}. {entity['text']} ({entity['source']} - {entity['field']})")


def main():
    print("开始抽取实体名称...\n")

    all_entities = []

    # 从各数据源抽取名称字段的值
    all_entities.extend(extract_from_entity_info())
    all_entities.extend(extract_from_fund_data())
    all_entities.extend(extract_from_indicator_info())
    all_entities.extend(extract_from_graph())

    print(f"\n总共抽取了 {len(all_entities)} 条记录")

    # 保存数据集
    save_dataset(all_entities, deduplicated=True)

    print("\n抽取完成!")


if __name__ == "__main__":
    main()
