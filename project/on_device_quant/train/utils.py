"""
工具函数模块

包含：
  - 余弦退火学习率调度器
  - JSON 日志记录器
  - Checkpoint 保存/加载
  - 早停控制器
"""
import math
import time
import json
import os
from pathlib import Path


def get_cosine_lr(step, warmup, total, base_lr, min_lr=1e-5):
    """
    余弦退火学习率调度（带线性预热）

    step 0 → warmup:  线性增长 0 → base_lr
    step warmup → total:  余弦衰减 base_lr → min_lr
    """
    if step < warmup:
        # 线性预热阶段
        return base_lr * (step + 1) / warmup
    # 余弦退火阶段
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Logger:
    """
    JSON 格式训练日志记录器

    记录每个标量（loss, ppl, lr 等）的 step 和 value，
    支持断点续写，方便后续可视化。
    """

    def __init__(self, log_path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 如果日志已存在，加载历史记录（支持续写）
        self.entries = []
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.entries = json.load(f)

    def log(self, entry):
        """写入一条日志记录"""
        entry["timestamp"] = time.time()
        self.entries.append(entry)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=1)

    def scalar(self, key, value, step):
        """记录一个标量值"""
        self.log({"key": key, "value": value, "step": step})


def save_checkpoint(model, optimizer, scaler, step, loss, path):
    """
    保存训练 checkpoint

    包含：模型权重、优化器状态、AMP scaler 状态、当前步数和 loss
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "loss": loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """
    加载训练 checkpoint

    返回: (start_step, last_loss)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0), ckpt.get("loss", 0)


class EarlyStopping:
    """
    早停控制器

    当验证集 loss 连续 patience 步没有改善时，触发停止。
    """

    def __init__(self, patience=500, min_delta=0.001):
        self.patience = patience    # 最多容忍多少步无改善
        self.min_delta = min_delta  # 最小改善量
        self.best = float("inf")    # 历史最佳 loss
        self.counter = 0            # 无改善步数计数

    def step(self, val_loss):
        """
        检查是否应该停止训练

        返回: True=应该停止, False=继续训练
        """
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
