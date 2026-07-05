"""
训练配置模块

采用 dataclass 管理模型、数据、训练三组参数，
支持 JSON 序列化保存/加载，方便实验复现。
"""
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """模型结构配置"""
    vocab_size: int = 4096       # BPE 词表大小
    dim: int = 512               # 隐藏层维度
    num_layers: int = 4          # 混合注意力层数
    num_heads: int = 8           # SWA 多头注意力头数
    head_dim: int = 64           # 每个注意力头的维度
    ffn_mult: int = 4            # FFN 扩展倍数
    dropout: float = 0.0         # Dropout 比率（0 表示不使用）
    max_seq_len: int = 512       # 最大序列长度（位置编码上限）
    weight_tie: bool = True      # 是否共享 embedding 和 LM head 权重

    @property
    def swa_dim(self):
        """SWA 注意力的总维度 = 头数 × 头维度"""
        return self.num_heads * self.head_dim


@dataclass
class DataConfig:
    """数据加载配置"""
    data_dir: str = ""           # 数据根目录
    coco_zip: str = ""           # COCO annotations zip 路径
    tokenizer_path: str = ""     # BPE tokenizer 保存路径
    seq_len: int = 512           # 训练序列长度
    batch_size: int = 8          # 批大小
    num_workers: int = 0         # DataLoader 工作进程数
    phase: str = "text"          # 训练阶段：text / multimodal
    max_captions: int = 0        # 最大 caption 数量（0=全部）
    min_caps_per_image: int = 2  # 多模态序列最少 caption 数


@dataclass
class TrainConfig:
    """训练超参数配置"""
    steps: int = 10000           # 总训练步数
    lr: float = 3e-4             # 学习率峰值
    weight_decay: float = 0.1    # 权重衰减
    warmup_steps: int = 200      # 学习率预热步数
    grad_clip: float = 1.0       # 梯度裁剪范数
    grad_accum: int = 1          # 梯度累积步数（等效放大 batch）
    log_every: int = 50          # 每 N 步打印训练日志
    eval_every: int = 500        # 每 N 步做一次验证
    ckpt_every: int = 1000       # 每 N 步保存 checkpoint
    val_batches: int = 20        # 验证时取多少个 batch
    use_amp: bool = True         # 是否使用混合精度训练（FP16）
    seed: int = 42               # 随机种子


@dataclass
class RunConfig:
    """运行总配置，聚合模型/数据/训练三部分"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "checkpoints"   # checkpoint 输出目录
    exp_name: str = "edge_transformer" # 实验名称（子目录名）
    device: str = "cuda"              # 训练设备
    resume_from: str = ""             # 恢复训练的 checkpoint 路径

    def save(self, path):
        """保存配置为 JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_dict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """从 JSON 加载配置"""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return _from_dict(cls, d)


def _to_dict(obj):
    """将 dataclass 递归转为字典"""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__ if getattr(obj, k) is not None}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _from_dict(cls, d):
    """从字典递归构建 dataclass"""
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        if k in fields:
            ft = fields[k].type
            if hasattr(ft, "__dataclass_fields__"):
                kwargs[k] = _from_dict(ft, v)
            else:
                kwargs[k] = v
    return cls(**kwargs)
