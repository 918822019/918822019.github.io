from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    model_path: str = "data/models/Ling-mini-base-2.0"
    device: str = "cpu"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 4
    max_prompt_len: int = 64
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 20
    hidden_size: int = 2048
    output_dir: str = str(Path(__file__).parent / "output")
    prompts: list = field(default_factory=lambda: [
        "def fibonacci(n):",
        "def merge_sort(arr):",
    ])
