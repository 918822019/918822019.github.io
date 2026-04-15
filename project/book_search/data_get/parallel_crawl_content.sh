#!/bin/bash

# 多进程并行爬取章节内容脚本
# 用法: bash parallel_crawl_content.sh [--num-procs 4] [--start 1] [--end 10000] [--concurrency 25] [--batch-size 100] [--db-path data/books.db]

set -e

# 默认参数
NUM_PROCS=4
START=1
END=10000
CONCURRENCY=25
BATCH_SIZE=100
DB_PATH="data/books.db"
MIN_REQUEST_INTERVAL=0.03
REQUEST_JITTER=0.05
CHAPTER_PROGRESS_EVERY=50
TIMEOUT=20
RETRIES=5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-procs)
            NUM_PROCS="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --min-request-interval)
            MIN_REQUEST_INTERVAL="$2"
            shift 2
            ;;
        --request-jitter)
            REQUEST_JITTER="$2"
            shift 2
            ;;
        --chapter-progress-every)
            CHAPTER_PROGRESS_EVERY="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --retries)
            RETRIES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 计算每个进程的范围
TOTAL_RANGE=$((END - START + 1))
RANGE_PER_PROC=$((TOTAL_RANGE / NUM_PROCS))

if [ $RANGE_PER_PROC -lt 1 ]; then
    RANGE_PER_PROC=1
    NUM_PROCS=$TOTAL_RANGE
fi

echo "================================================================"
echo "并行爬取章节内容"
echo "总范围: $START - $END (共 $TOTAL_RANGE 本书)"
echo "进程数: $NUM_PROCS"
echo "每进程范围: $RANGE_PER_PROC 本书"
echo "每进程并发: $CONCURRENCY"
echo "数据库: $DB_PATH"
echo "================================================================"
echo ""

# 创建临时目录存放进程 PID 和日志
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

# 启动多个进程
pids=()
for ((i = 0; i < NUM_PROCS; i++)); do
    proc_start=$((START + i * RANGE_PER_PROC))
    if [ $i -eq $((NUM_PROCS - 1)) ]; then
        # 最后一个进程处理到 END
        proc_end=$END
    else
        proc_end=$((proc_start + RANGE_PER_PROC - 1))
    fi
    
    # 跳过范围已为空的进程
    if [ $proc_start -gt $END ]; then
        continue
    fi
    
    echo "[进程 $((i+1))/$NUM_PROCS] 启动爬取: $proc_start - $proc_end"
    
    # 后台启动爬虫进程，将日志重定向到文件
    python main.py crawl-content \
        --start $proc_start \
        --end $proc_end \
        --concurrency $CONCURRENCY \
        --batch-size $BATCH_SIZE \
        --db-path "$DB_PATH" \
        --min-request-interval $MIN_REQUEST_INTERVAL \
        --request-jitter $REQUEST_JITTER \
        --chapter-progress-every $CHAPTER_PROGRESS_EVERY \
        --timeout $TIMEOUT \
        --retries $RETRIES \
        > "$WORK_DIR/proc_${i}.log" 2>&1 &
    
    pids+=($!)
done

echo ""
echo "所有进程已启动，PID: ${pids[@]}"
echo "监控日志: tail -f $WORK_DIR/proc_*.log"
echo ""

# 等待所有进程完成
failed=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    proc_start=$((START + i * RANGE_PER_PROC))
    if [ $i -eq $((NUM_PROCS - 1)) ]; then
        proc_end=$END
    else
        proc_end=$((proc_start + RANGE_PER_PROC - 1))
    fi
    
    if wait $pid 2>/dev/null; then
        echo "✓ [进程 $((i+1))] ($proc_start-$proc_end) 完成"
    else
        echo "✗ [进程 $((i+1))] ($proc_start-$proc_end) 失败"
        failed=$((failed + 1))
    fi
done

echo ""
echo "================================================================"
if [ $failed -eq 0 ]; then
    echo "✓ 所有进程完成！共 $NUM_PROCS 个进程，0 个失败"
    echo ""
    python main.py stats --db-path "$DB_PATH"
else
    echo "✗ 有 $failed 个进程失败，请检查日志："
    for i in "${!pids[@]}"; do
        echo "  - $WORK_DIR/proc_${i}.log"
    done
    exit 1
fi
echo "================================================================"
