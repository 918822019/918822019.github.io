import math, time, json, os
from pathlib import Path


def get_cosine_lr(step, warmup, total, base_lr, min_lr=1e-5):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Logger:
    def __init__(self, log_path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []
        if self.path.exists():
            with open(self.path) as f:
                self.entries = json.load(f)

    def log(self, entry):
        entry["timestamp"] = time.time()
        self.entries.append(entry)
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=1)

    def scalar(self, key, value, step):
        self.log({"key": key, "value": value, "step": step})


def save_checkpoint(model, optimizer, scaler, step, loss, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "loss": loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0), ckpt.get("loss", 0)


class EarlyStopping:
    def __init__(self, patience=500, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
