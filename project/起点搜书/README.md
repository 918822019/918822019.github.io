# 起点搜书

本模块实现基于用户查询的书籍推荐系统，包含查询处理、数据召回、推荐理由生成和结果展示等功能。

## 目录结构

- data/books.json 书籍数据
- query_processor.py 查询预处理
- rag_retriever.py 数据召回
- app.py 推荐API服务

## 快速开始

1. 准备书籍数据到 data/books.json
2. 运行 app.py 启动服务
3. 前端通过 API 获取推荐结果
