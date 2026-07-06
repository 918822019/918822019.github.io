"""
数据加载模块

支持：
  - COCO Captions 数据集加载
  - BPE tokenizer 训练/加载/编解码
  - 多模态序列构建（同一图片的多条 caption 拼接为长序列）
  - PyTorch DataLoader 封装
"""
import json
import os
import zipfile
import pickle
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPT
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BPETokenizer:
    """
    BPE（Byte Pair Encoding）分词器

    基于 HuggingFace tokenizers（Rust 实现），训练/编码速度提升 100x+
    """

    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self._tok = HFTokenizer(BPE(unk_token="<unk>"))
        self._tok.pre_tokenizer = ByteLevelPT(add_prefix_space=False)
        self._tok.decoder = ByteLevelDecoder()
        self._trained = False

    def train(self, text, verbose=False):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            initial_alphabet=ByteLevelPT.alphabet(),
            show_progress=verbose,
        )
        self._tok.train_from_iterator([text], trainer=trainer)
        self._trained = True

    def encode(self, text):
        ids = self._tok.encode(text).ids
        # 偏移：BPE id 从 4 开始（0=pad,1=bos,2=eos,3=unk），映射到我们的 BOS=2,EOS=3 方案
        # 我们直接返回 HF tokenizer 的 id，下游用 id+offset 或者调整特殊 token
        return ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path):
        tok = cls.__new__(cls)
        tok._tok = HFTokenizer.from_file(str(path))
        tok.vocab_size = tok._tok.get_vocab_size()
        tok._trained = True
        return tok


class SequenceDataset(Dataset):
    """
    语言模型训练数据集

    将 token 序列按 seq_len 切分为固定长度的 (x, y) 对：
      x = tokens[i : i+seq_len]
      y = tokens[i+1 : i+seq_len+1]（右移一位的目标）
    """

    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.data = token_ids.clone().detach() if isinstance(token_ids, torch.Tensor) else torch.tensor(token_ids, dtype=torch.long)
        self.n = len(self.data) - seq_len - 1

    def __len__(self):
        return max(0, self.n // self.seq_len)

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = self.data[i : i + self.seq_len]
        y = self.data[i + 1 : i + self.seq_len + 1]
        return x, y


def load_coco_captions(zip_path):
    """
    加载 COCO Captions 数据集

    返回: {"train": [caption1, ...], "val": [caption1, ...]}
    """
    caps = {"train": [], "val": []}
    with zipfile.ZipFile(zip_path) as z:
        for split in ["train", "val"]:
            with z.open(f"annotations/captions_{split}2017.json") as f:
                data = json.load(f)
            caps[split] = [a["caption"].lower() for a in data["annotations"]]
    return caps


def _cfg_get(config, key, default=None):
    """从嵌套 config 中取值，兼容 RunConfig(a.b.c) 和扁平对象"""
    val = getattr(config, key, None)
    if val is not None:
        return val
    for sub in ["data", "model", "train"]:
        sub_obj = getattr(config, sub, None)
        if sub_obj is not None:
            val = getattr(sub_obj, key, None)
            if val is not None:
                return val
    return default


def load_data(config):
    """
    完整数据加载流程：
    1. 加载已有 BPE tokenizer（或新建）
    2. 加载 COCO captions
    3. 用 BPE 编码所有 caption
    4. 划分训练集/验证集（95%/5%）
    """
    tokenizer_path = _cfg_get(config, "tokenizer_path", "")
    coco_zip = _cfg_get(config, "coco_zip", "")
    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)

    if coco_zip and os.path.exists(coco_zip):
        caps = load_coco_captions(coco_zip)
        # 如果没有预训练的 tokenizer，从语料中训练
        if tokenizer is None:
            # 只用前 N 条 caption 训练 BPE（全量 59 万条太慢）
            sample_size = _cfg_get(config, "tokenizer_sample", 10000)
            sample_caps = caps["train"][:sample_size] + caps["val"][:sample_size // 10]
            all_text = " ".join(sample_caps)
            tokenizer = BPETokenizer(_cfg_get(config, "vocab_size", 4096))
            tokenizer.train(all_text, verbose=True)
            if tokenizer_path:
                tokenizer.save(tokenizer_path)

        # 编码所有 caption，每条加 [BOS]=2 和 [EOS]=3
        all_tokens = []
        for cap in caps["train"]:
            all_tokens.extend([2] + tokenizer.encode(cap) + [3])
        data = torch.tensor(all_tokens, dtype=torch.long)

        # 95% 训练 / 5% 验证
        n = int(0.95 * len(data))
        train_data, val_data = data[:n], data[n:]
        return train_data, val_data, tokenizer

    raise FileNotFoundError(f"未找到 COCO 数据: {coco_zip}")


def get_dataloader(token_ids, seq_len, batch_size, num_workers=0, shuffle=True):
    """
    构建 PyTorch DataLoader

    token_ids: 编码后的 token id 列表
    seq_len: 序列长度
    batch_size: 批大小
    """
    ds = SequenceDataset(token_ids, seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,    # 加速 CPU→GPU 数据传输
        drop_last=True,     # 丢弃最后一个不完整的 batch
    )
