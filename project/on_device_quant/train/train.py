"""
EdgeTransformer 训练流水线

使用方法：
    python train.py                          # 默认配置，COCO 数据
    python train.py --steps 20000 --lr 5e-4  # 覆盖超参数
    python train.py --resume checkpoints/edge_transformer/last.pt  # 恢复训练
    python train.py --dim 768 --num_layers 6 --compile  # 大模型 + torch.compile

特性：
  - 混合注意力架构（CSA + HCA + SWA 三分支金字塔）
  - RMSNorm 归一化（可选 LayerNorm 兼容旧 checkpoint）
  - AMP 混合精度训练（FP16 + FP32，HybridBlock 内强制 FP32）
  - torch.compile 加速（编译 GEMM 密集部分，--compile 开启）
  - 梯度累积（等效放大 batch size）
  - 余弦退火学习率调度（带线性预热）
  - Checkpoint 自动保存（best/last/定期）
  - JSON 日志记录
"""
import os
import sys

# 默认使用 HuggingFace 镜像，加速国内访问（用户可 export HF_ENDPOINT 覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import time
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import RunConfig, ModelConfig, DataConfig, TrainConfig
from model import EdgeTransformer, count_params
from data import load_data, get_dataloader
from utils import get_cosine_lr, Logger, save_checkpoint, load_checkpoint


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="EdgeTransformer 训练")
    # 运行配置
    p.add_argument("--config", type=str, default="", help="配置 JSON 路径")
    p.add_argument("--exp_name", type=str, default="edge_transformer", help="实验名称")
    p.add_argument("--output_dir", type=str, default="checkpoints", help="输出目录")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default="", help="恢复训练的 checkpoint 路径")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    # 模型配置
    p.add_argument("--dim", type=int, default=512, help="隐藏层维度")
    p.add_argument("--num_layers", type=int, default=4, help="Transformer 层数")
    p.add_argument("--num_heads", type=int, default=8, help="SWA 多头注意力头数")
    p.add_argument("--head_dim", type=int, default=64, help="每个头的维度")
    p.add_argument("--seq_len", type=int, default=512, help="序列长度")
    p.add_argument("--batch_size", type=int, default=8, help="批大小")

    # 训练配置
    p.add_argument("--steps", type=int, default=10000, help="总训练步数")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    p.add_argument("--warmup_steps", type=int, default=200, help="预热步数")
    p.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    p.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--use_amp", action="store_true", default=True, help="使用混合精度")
    p.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    p.add_argument("--norm_type", type=str, default="rms", choices=["rms", "layernorm"], help="归一化层类型")
    p.add_argument("--compile", action="store_true", help="启用 torch.compile")
    p.add_argument("--compile_mode", type=str, default="default", help="编译模式")

    # 日志与保存
    p.add_argument("--log_every", type=int, default=50, help="每 N 步打印日志")
    p.add_argument("--eval_every", type=int, default=500, help="每 N 步验证")
    p.add_argument("--ckpt_every", type=int, default=1000, help="每 N 步保存 checkpoint")

    # 数据
    p.add_argument("--coco_zip", type=str, default="", help="COCO zip 路径")
    p.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer.pkl", help="BPE tokenizer 路径")
    p.add_argument("--data_source", type=str, default="coco", choices=["coco", "skypile"], help="数据源类型")
    p.add_argument("--hf_dataset", type=str, default="Skywork/SkyPile-150B", help="HuggingFace dataset name")
    p.add_argument("--hf_train_split", type=str, default="train", help="HF train split")
    p.add_argument("--hf_val_split", type=str, default="", help="HF val split（空则从 train 取样本）")
    p.add_argument("--max_samples", type=int, default=0, help="最大样本数（0=无限）")
    p.add_argument("--max_size_gb", type=float, default=0, help="最大读取数据量 GB（0=不限制）")
    p.add_argument("--tokenizer_sample", type=int, default=10000, help="BPE 训练采样条数")

    args = p.parse_args()
    if args.no_amp:
        args.use_amp = False
    return args


def build_config(args):
    """从命令行参数构建运行配置"""
    model = ModelConfig(
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        max_seq_len=args.seq_len,
        norm_type=args.norm_type,
    )
    data = DataConfig(
        coco_zip=args.coco_zip,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        data_source=args.data_source,
        hf_dataset=args.hf_dataset,
        hf_train_split=args.hf_train_split,
        hf_val_split=args.hf_val_split,
        max_samples=args.max_samples,
        max_size_gb=args.max_size_gb,
        tokenizer_sample=args.tokenizer_sample,
    )
    train = TrainConfig(
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        grad_accum=args.grad_accum,
        log_every=args.log_every,
        eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        use_amp=args.use_amp,
        seed=args.seed,
        compile=args.compile,
        compile_mode=args.compile_mode,
    )
    return RunConfig(
        model=model,
        data=data,
        train=train,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        device=args.device,
        resume_from=args.resume,
    )


def find_coco_zip():
    """
    自动查找 COCO 数据集 zip 文件

    按优先级依次尝试多个候选路径（绝对路径 → 相对路径），
    找到第一个存在的即返回，未找到返回空字符串。
    """
    candidates = [
        # 绝对路径：macOS 桌面项目目录
        os.path.expanduser("~/Desktop/918822019.github.io/data/coco/PAI/COCO2017/annotations_trainval2017.zip"),
        # 相对路径：从不同工作目录运行时的回退
        "data/coco/PAI/COCO2017/annotations_trainval2017.zip",
        "../data/coco/PAI/COCO2017/annotations_trainval2017.zip",
        "../../data/coco/PAI/COCO2017/annotations_trainval2017.zip",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return ""


@torch.no_grad()
def evaluate(model, val_loader, device, vocab_size, num_batches=20, is_streaming=False):
    """
    验证：在验证集上计算平均 loss

    model: 待评估模型
    val_loader: 验证集 DataLoader
    num_batches: 取多少个 batch 计算平均（控制验证时间）
    is_streaming: 流式模式需用持久迭代器取 batch
    """
    model.eval()
    total_loss, count = 0.0, 0
    if is_streaming:
        val_iter = iter(val_loader)
        for i in range(num_batches):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
            count += 1
    else:
        for i, (x, y) in enumerate(val_loader):
            if i >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def train(config):
    """
    主训练循环

    流程：
      1. 设置随机种子和设备
      2. 加载数据和 tokenizer
      3. 构建模型、优化器、AMP scaler
      4. 训练循环：前向 → 损失 → 反向 → 梯度裁剪 → 参数更新
      5. 定期验证和保存 checkpoint
    """
    # 固定随机种子
    torch.manual_seed(config.train.seed)
    device = torch.device(config.device)
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载数据和 tokenizer
    print("加载数据...")
    from data import load_streaming_data
    if config.data.data_source == "skypile":
        print(f"数据源: {config.data.hf_dataset} (streaming)")
        train_loader, val_loader, tokenizer = load_streaming_data(config.data)
        vocab_size = tokenizer.vocab_size
        config.model.vocab_size = vocab_size
        print(f"词表: {vocab_size} | 流式模式（不预加载全部数据）")
        is_streaming = True
    else:
        # COCO 路径：全量载入内存
        if not config.data.coco_zip:
            config.data.coco_zip = find_coco_zip()
        if not config.data.coco_zip:
            print("错误: 未找到 COCO 数据，请使用 --coco_zip <路径> 指定")
            sys.exit(1)
        print(f"数据: {config.data.coco_zip}")
        train_ids, val_ids, tokenizer = load_data(config.data)
        vocab_size = tokenizer.vocab_size
        config.model.vocab_size = vocab_size
        print(f"词表: {vocab_size} | 训练 token: {len(train_ids):,} | 验证 token: {len(val_ids):,}")
        train_loader = get_dataloader(train_ids, config.data.seq_len, config.data.batch_size, shuffle=True)
        val_loader = get_dataloader(val_ids, config.data.seq_len, config.data.batch_size, shuffle=False)
        is_streaming = False

    # 构建模型
    print("构建模型...")
    model = EdgeTransformer(
        vocab_size=config.model.vocab_size,
        dim=config.model.dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        ffn_mult=config.model.ffn_mult,
        max_seq_len=config.model.max_seq_len,
        weight_tie=config.model.weight_tie,
        norm_type=config.model.norm_type,
    ).to(device)
    print(f"参数量: {count_params(model):,}")
    print(f"归一化: {config.model.norm_type}")

    # torch.compile 加速（编译 GEMM 密集部分，linear_attn 循环回退 eager）
    if config.train.compile and device.type == "cuda":
        print(f"torch.compile ({config.train.compile_mode})...")
        model = torch.compile(model, mode=config.train.compile_mode)

    # 优化器：AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.95),
    )
    # AMP 混合精度缩放器（仅 CUDA 可用）
    scaler = torch.amp.GradScaler(enabled=config.train.use_amp) if device.type == "cuda" else None
    start_step = 0

    # 创建 checkpoint 目录并保存配置
    ckpt_dir = Path(config.output_dir) / config.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(ckpt_dir / "config.json"))

    # 恢复训练（如果指定）
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"从 {config.resume_from} 恢复训练...")
        start_step, _ = load_checkpoint(config.resume_from, model, optimizer, scaler)
        print(f"已恢复到第 {start_step} 步")

    # 初始化日志
    logger = Logger(str(ckpt_dir / "log.json"))
    best_val_loss = float("inf")
    t0 = time.time()
    total_steps = config.train.steps

    # 打印训练配置
    print(f"\n{'=' * 60}")
    print(f"训练 {total_steps} 步 | dim={config.model.dim} | 层数={config.model.num_layers} | 头数={config.model.num_heads}")
    print(f"batch={config.data.batch_size} x seq={config.data.seq_len} | 混合精度={'开启' if config.train.use_amp else '关闭'}")
    print(f"{'=' * 60}\n")

    # ── 训练主循环 ──
    # 流式模式用持久迭代器（创建一次，逐步取 batch，不重建）
    train_iter = iter(train_loader) if is_streaming else None
    for step in range(start_step, total_steps):
        # 更新学习率（余弦退火 + 预热）
        lr = get_cosine_lr(step, config.train.warmup_steps, total_steps, config.train.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        micro_loss = 0.0

        # 梯度累积：将大 batch 拆成多个 micro-batch 依次前向反向
        for micro_step in range(config.train.grad_accum):
            if is_streaming:
                # 流式模式：从持久迭代器取 batch
                x, y = next(train_iter)
            else:
                # COCO 模式：每步重建迭代器（简单但非最优）
                x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)

            # 混合精度前向传播
            with torch.amp.autocast(device_type=device.type, enabled=config.train.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)) / config.train.grad_accum

            # 反向传播（带 AMP 缩放）
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_loss += loss.item()

        # 梯度裁剪 + 参数更新
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ── 定期打印训练日志 ──
        if step % config.train.log_every == 0:
            ppl = math.exp(min(micro_loss * config.train.grad_accum, 20))
            dt = time.time() - t0
            tps = (step + 1 - start_step) * config.data.batch_size * config.data.seq_len / dt if dt > 0 else 0
            print(f"step {step:6d} | loss {micro_loss * config.train.grad_accum:.4f} | ppl {ppl:8.2f} | lr {lr:.2e} | {dt:.0f}s | {tps:.0f} tok/s")
            logger.scalar("train_loss", micro_loss * config.train.grad_accum, step)
            logger.scalar("train_ppl", ppl, step)
            logger.scalar("lr", lr, step)

        # ── 定期验证 ──
        if step > 0 and step % config.train.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, vocab_size, config.train.val_batches, is_streaming)
            val_ppl = math.exp(min(val_loss, 20))
            tag = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scaler, step, val_loss, str(ckpt_dir / "best.pt"))
                tag = " ★ 最佳"
            print(f"  [验证] loss {val_loss:.4f} | ppl {val_ppl:8.2f}{tag}")
            logger.scalar("val_loss", val_loss, step)
            logger.scalar("val_ppl", val_ppl, step)

        # ── 定期保存 checkpoint ──
        if step > 0 and step % config.train.ckpt_every == 0:
            save_checkpoint(model, optimizer, scaler, step, micro_loss, str(ckpt_dir / f"step{step}.pt"))

    # 保存最终 checkpoint
    save_checkpoint(model, optimizer, scaler, total_steps, 0, str(ckpt_dir / "last.pt"))
    print(f"\n{'=' * 60}")
    print(f"训练完成! 耗时 {time.time() - t0:.0f}s | 最佳验证 loss: {best_val_loss:.4f} | ppl: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"Checkpoint 目录: {ckpt_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    train(config)
