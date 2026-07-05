"""
EdgeTransformer Training Pipeline
Usage:
    python train.py                          # default config, COCO data
    python train.py --steps 20000 --lr 5e-4  # override params
    python train.py --resume checkpoints/edge_transformer/last.pt
"""
import argparse, os, sys, time, math, json
from pathlib import Path

import torch
import torch.nn.functional as F

from config import RunConfig, ModelConfig, DataConfig, TrainConfig
from model import EdgeTransformer, count_params
from data import load_data, get_dataloader, BPETokenizer
from utils import get_cosine_lr, Logger, save_checkpoint, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="EdgeTransformer Training")
    p.add_argument("--config", type=str, default="", help="config json path")
    p.add_argument("--exp_name", type=str, default="edge_transformer")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--coco_zip", type=str, default="")
    p.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer.pkl")

    args = p.parse_args()
    if args.no_amp:
        args.use_amp = False
    return args


def build_config(args):
    model = ModelConfig(
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        max_seq_len=args.seq_len,
    )
    data = DataConfig(
        coco_zip=args.coco_zip,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
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
    candidates = [
        os.path.expanduser("~/Desktop/918822019.github.io/data/coco/PAI/COCO2017/annotations_trainval2017.zip"),
        "data/coco/PAI/COCO2017/annotations_trainval2017.zip",
        "../data/coco/PAI/COCO2017/annotations_trainval2017.zip",
        "../../data/coco/PAI/COCO2017/annotations_trainval2017.zip",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return ""


@torch.no_grad()
def evaluate(model, val_loader, device, vocab_size, num_batches=20):
    model.eval()
    total_loss, count = 0.0, 0
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
    torch.manual_seed(config.train.seed)
    device = torch.device(config.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    if not config.data.coco_zip:
        config.data.coco_zip = find_coco_zip()
    if not config.data.coco_zip:
        print("ERROR: COCO data not found. Use --coco_zip <path>")
        sys.exit(1)
    print(f"Data: {config.data.coco_zip}")

    print("Loading data...")
    train_ids, val_ids, tokenizer = load_data(config.data)
    vocab_size = len(tokenizer.vocab)
    config.model.vocab_size = vocab_size
    print(f"Vocab: {vocab_size} | Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

    train_loader = get_dataloader(train_ids, config.data.seq_len, config.data.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ids, config.data.seq_len, config.data.batch_size, shuffle=False)

    print("Building model...")
    model = EdgeTransformer(
        vocab_size=config.model.vocab_size,
        dim=config.model.dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        ffn_mult=config.model.ffn_mult,
        max_seq_len=config.model.max_seq_len,
        weight_tie=config.model.weight_tie,
    ).to(device)
    print(f"Params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler(enabled=config.train.use_amp) if device.type == "cuda" else None
    start_step = 0

    ckpt_dir = Path(config.output_dir) / config.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(ckpt_dir / "config.json"))

    if config.resume_from and os.path.exists(config.resume_from):
        print(f"Resuming from {config.resume_from}")
        start_step, _ = load_checkpoint(config.resume_from, model, optimizer, scaler)
        print(f"Resumed at step {start_step}")

    logger = Logger(str(ckpt_dir / "log.json"))
    best_val_loss = float("inf")
    t0 = time.time()
    total_steps = config.train.steps

    print(f"\n{'=' * 60}")
    print(f"Training {total_steps} steps | dim={config.model.dim} | layers={config.model.num_layers} | heads={config.model.num_heads}")
    print(f"batch={config.data.batch_size} x seq={config.data.seq_len} | AMP={'ON' if config.train.use_amp else 'OFF'}")
    print(f"{'=' * 60}\n")

    for step in range(start_step, total_steps):
        lr = get_cosine_lr(step, config.train.warmup_steps, total_steps, config.train.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        micro_loss = 0.0
        for micro_step in range(config.train.grad_accum):
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=config.train.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)) / config.train.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_loss += loss.item()

        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % config.train.log_every == 0:
            ppl = math.exp(min(micro_loss * config.train.grad_accum, 20))
            dt = time.time() - t0
            tps = (step + 1 - start_step) * config.data.batch_size * config.data.seq_len / dt if dt > 0 else 0
            print(f"step {step:6d} | loss {micro_loss * config.train.grad_accum:.4f} | ppl {ppl:8.2f} | lr {lr:.2e} | {dt:.0f}s | {tps:.0f} tok/s")
            logger.scalar("train_loss", micro_loss * config.train.grad_accum, step)
            logger.scalar("train_ppl", ppl, step)
            logger.scalar("lr", lr, step)

        if step > 0 and step % config.train.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, vocab_size, config.train.val_batches)
            val_ppl = math.exp(min(val_loss, 20))
            tag = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scaler, step, val_loss, str(ckpt_dir / "best.pt"))
                tag = " * best"
            print(f"  [val] loss {val_loss:.4f} | ppl {val_ppl:8.2f}{tag}")
            logger.scalar("val_loss", val_loss, step)
            logger.scalar("val_ppl", val_ppl, step)

        if step > 0 and step % config.train.ckpt_every == 0:
            save_checkpoint(model, optimizer, scaler, step, micro_loss, str(ckpt_dir / f"step{step}.pt"))

    save_checkpoint(model, optimizer, scaler, total_steps, 0, str(ckpt_dir / "last.pt"))
    print(f"\n{'=' * 60}")
    print(f"Done! {time.time() - t0:.0f}s | Best val loss: {best_val_loss:.4f} | ppl: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    train(config)
