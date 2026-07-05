"""Configuration for the MTP Router Consistency Analysis pipeline.

All model and analysis parameters are defined here.
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    # Model
    model_path: str = "data/models/Ling-mini-base-2.0"
    device: str = "cpu"
    torch_dtype: str = "bfloat16"

    # Generation (currently unused - single forward pass only)
    max_new_tokens: int = 4
    max_prompt_len: int = 64

    # MoE architecture
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 20
    hidden_size: int = 2048

    # Output
    output_dir: str = str(Path(__file__).resolve().parent.parent / "output")

    # Test prompts
    prompts: list = field(default_factory=lambda: [
        "def fibonacci(n):",
        "def merge_sort(arr):",
    ])
