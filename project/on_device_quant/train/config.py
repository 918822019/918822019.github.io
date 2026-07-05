from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 64
    ffn_mult: int = 4
    dropout: float = 0.0
    max_seq_len: int = 512
    weight_tie: bool = True

    @property
    def swa_dim(self):
        return self.num_heads * self.head_dim


@dataclass
class DataConfig:
    data_dir: str = ""
    coco_zip: str = ""
    tokenizer_path: str = ""
    seq_len: int = 512
    batch_size: int = 8
    num_workers: int = 0
    phase: str = "text"
    max_captions: int = 0
    min_caps_per_image: int = 2


@dataclass
class TrainConfig:
    steps: int = 10000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    grad_clip: float = 1.0
    grad_accum: int = 1
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 1000
    val_batches: int = 20
    use_amp: bool = True
    seed: int = 42


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "checkpoints"
    exp_name: str = "edge_transformer"
    device: str = "cuda"
    resume_from: str = ""

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(_to_dict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return _from_dict(cls, d)


def _to_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in obj.__dataclass_fields__.items() if getattr(obj, k) is not None}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


def _from_dict(cls, d):
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
