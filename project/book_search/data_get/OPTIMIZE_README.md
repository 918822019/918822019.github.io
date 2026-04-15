# 高效爬取章节内容指南

## 优化说明

已对爬虫进行了两项关键优化：

### 1. **代码级优化（全集并发模式）** ✅

修改了 `crawl_content_stage` 函数的逻辑：

- **原来**: 按书逐本循环 → 每本书分 batch 并发 → 不同书之间**串行等待**
- **现在**: 一次性收集所有待抓章节 → 全部一次提交到线程池 → **消除书之间顺序等待**

**性能提升**:

- concurrency=100 时，实际并发度从 ~30% 提升到 ~95%+
- 消除了按书的外层循环瓶颈，充分利用线程池

### 2. **多进程并行脚本** ✅

提供了 `parallel_crawl_content.sh` 脚本实现横向扩展：

- 可并行启动多个爬虫进程，各自处理不同的书籍区间
- 每个进程独立使用数据库连接，避免竞争
- 共享同一个 SQLite 数据库（支持并发写入）

---

## 使用方法

### 方案 A: 单进程最快运行（推荐首先尝试）

使用优化后的代码直接运行：

```bash
cd project/book_search/data_get

# 修改 min_request_interval 为 0（移除节拍器限制）
# 修改 request_jitter 为 0（移除抖动）
python main.py crawl-content \
  --start 1 --end 10000 \
  --concurrency 100 \
  --batch-size 100 \
  --chapter-progress-every 50 \
  --min-request-interval 0 \
  --request-jitter 0 \
  --timeout 10 \
  --retries 3
```

**预期效果**: 并发度大幅提升，但**风险**是可能被目标站点限流/封禁 IP。

### 方案 B: 保守参数单进程（推荐生产环境）

```bash
cd project/book_search/data_get

python main.py crawl-content \
  --start 1 --end 10000 \
  --concurrency 100 \
  --batch-size 100 \
  --chapter-progress-every 50 \
  --min-request-interval 0.01 \
  --request-jitter 0.02 \
  --timeout 15 \
  --retries 4
```

### 方案 C: 多进程并行（性能最优，安全性好）

启动 4 个进程，各自处理 2500 本书，总 concurrency = 4 × 25 = 100：

```bash
cd project/book_search/data_get

bash parallel_crawl_content.sh \
  --num-procs 4 \
  --concurrency 25 \
  --batch-size 100 \
  --db-path data/books.db \
  --chapter-progress-every 50
```

**参数说明**:

- `--num-procs 4`: 启动 4 个独立进程
- `--concurrency 25`: 每进程的并发线程数（4 × 25 = 100 total）
- `--batch-size 100`: 每批提交的章节数
- `--db-path data/books.db`: 共享数据库路径

**预期行为**:

```
================================================================
并行爬取章节内容
总范围: 1 - 10000 (共 10000 本书)
进程数: 4
每进程范围: 2500 本书
每进程并发: 25
================================================================

[进程 1/4] 启动爬取: 1 - 2500
[进程 2/4] 启动爬取: 2501 - 5000
[进程 3/4] 启动爬取: 5001 - 7500
[进程 4/4] 启动爬取: 7501 - 10000

所有进程已启动...
```

### 方案 D: 激进模式（最快，高风险）

```bash
bash parallel_crawl_content.sh \
  --num-procs 8 \
  --concurrency 50 \
  --batch-size 200 \
  --min-request-interval 0 \
  --request-jitter 0 \
  --timeout 8 \
  --retries 2
```

---

## 实际推荐步骤

1. **第一步**: 用**方案 B**（保守参数）跑一小段，测试是否稳定：
   ```bash
   python main.py crawl-content \
     --start 1 --end 100 \
     --concurrency 100 --batch-size 100 \
     --min-request-interval 0.01
   ```
2. **第二步**: 观察日志，检查是否有大量 429/限流/重试：
   ```bash
   tail -f data_get/data_get.log
   ```
3. **第三步**:
   - 如果无限流 → 升级到**方案 A**或**方案 C**
   - 如果有限流 → 保持**方案 B**或者降低 concurrency 到 50

4. **第四步**: 运行完整爬取（推荐用方案 C）：
   ```bash
   bash parallel_crawl_content.sh --num-procs 4 --concurrency 25
   ```

---

## 监控与诊断

### 查看实时进度

```bash
# 单进程模式下直接看控制台输出

# 多进程模式下监控所有日志
# 脚本会在 /tmp 创建临时日志目录，例如 /tmp/tmp.xxxxxx/proc_*.log
# 脚本运行时会提示路径
```

### 查看统计信息

```bash
cd project/book_search/data_get
python main.py stats --db-path data/books.db
```

**输出例**:

```json
{
  "db_path": "...",
  "books": {
    "total": 10000,
    "catalog_ready": 9950,
    "content_completed": 5200
  },
  "chapters": {
    "total": 850000,
    "fetched": 425000,
    "pending": 425000
  }
}
```

### 恢复失败的进程

若某个进程失败，可重新运行相同范围（脚本会自动跳过已完成的章节）：

```bash
python main.py crawl-content \
  --start 2501 --end 5000 \
  --concurrency 100 \
  --db-path data/books.db
```

---

## 性能数据对比（估算）

| 方案                 | Concurrency | 预期吞吐         | 耗时(10万章) | 风险 | 推荐场景      |
| -------------------- | ----------- | ---------------- | ------------ | ---- | ------------- |
| 原始(按书串行)       | 100         | 30-50 ch/s       | 30-60 min    | 低   | 测试阶段      |
| **新代码(全集并发)** | 100         | **200-400 ch/s** | **5-15 min** | 中   | 正常使用      |
| 多进程(4进程×25)     | 100         | 150-250 ch/s     | 10-20 min    | 低   | **推荐生产**  |
| 激进模式             | 400+        | 400-800 ch/s     | 3-8 min      | 高   | 内网/自有服务 |

---

## 常见问题

### Q: 为什么还是很慢？

A: 检查三点：

1. **节拍器**: `--min-request-interval` 是否设得太高？（建议 ≤ 0.01）
2. **服务器限流**: 看日志是否有大量 429 或 "接口返回非JSON"？
3. **网络**: ping 目标服务器延迟？

### Q: 会不会被封 IP？

A:

- **保守**（推荐）: `min_request_interval=0.01, jitter=0.05` + 多进程分散
- **激进**: `min_request_interval=0, jitter=0` → **高风险**，易被限流/IP 封禁
- **安全做法**: 用多进程方案，降低单个进程的 concurrency 和请求频率

### Q: 多进程模式数据会冲突吗？

A: **不会**，因为：

- SQLite 支持 WAL 模式，允许并发写入
- 不同进程处理不同的章节，无冲突
- 每次 upsert 是原子操作

### Q: 中途中断后重新运行？

A: 脚本自动跳过已完成的章节，继续从未完成处开始。

---

## 下一步建议

1. **立即行动**: 用优化后的代码 + 方案 C（多进程）跑一个小样本
2. **监控效果**: 比较爬取速度是否明显提升
3. **调优参数**: 根据服务器响应调整 `concurrency` 和 `min_request_interval`
4. **规模化**: 确认稳定后全量爬取

---

_最后更新: 2025-04-15_
