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


class BPETokenizer:
    """
    BPE（Byte Pair Encoding）分词器

    原理：从字节级别开始，反复合并最频繁的相邻 token 对，
    直到达到目标词表大小。兼顾字符级和词级分词的优点。
    """

    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size  # 目标词表大小
        self.merges = []              # 合并规则列表 [(id_a, id_b, new_id), ...]
        self.vocab = {}               # id → bytes 映射
        self.inverse_vocab = {}       # bytes → id 映射

    def train(self, text, verbose=False):
        """
        训练 BPE：从语料中学习合并规则

        text: 训练语料（字符串）
        verbose: 是否打印训练进度
        """
        # 转为字节序列
        tokens = list(text.encode("utf-8"))

        for i in range(self.vocab_size - 256):
            # 统计所有相邻 token 对的频率
            counts = Counter()
            for a, b in zip(tokens, tokens[1:]):
                counts[(a, b)] += 1
            if not counts:
                break

            # 找到最频繁的 token 对
            best = max(counts, key=counts.get)
            new_id = 256 + i  # 新 token id 从 256 开始

            # 记录合并规则
            self.merges.append((best[0], best[1], new_id))

            # 执行合并：将 best 对替换为 new_id
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == best[0] and tokens[j + 1] == best[1]:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

            if verbose and i % 100 == 0:
                print(f"  merge {i}: vocab {256 + i}, tokens {len(tokens)}")

        # 构建 vocab 映射
        self.vocab = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in self.merges:
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        """
        编码：字符串 → token id 列表
        按顺序应用所有合并规则
        """
        tokens = list(text.encode("utf-8"))
        for a, b, new_id in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        """解码：token id 列表 → 字符串"""
        return b"".join(self.vocab.get(i, b"?") for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        """保存 tokenizer 到文件"""
        with open(path, "wb") as f:
            pickle.dump({"merges": self.merges, "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path):
        """从文件加载 tokenizer"""
        with open(path, "rb") as f:
            d = pickle.load(f)
        tok = cls(d["vocab_size"])
        tok.merges = d["merges"]
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in tok.merges:
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
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


def load_data(config):
    """
    完整数据加载流程：
    1. 加载已有 BPE tokenizer（或新建）
    2. 加载 COCO captions
    3. 用 BPE 编码所有 caption
    4. 划分训练集/验证集（95%/5%）
    """
    tokenizer = None
    if config.tokenizer_path and os.path.exists(config.tokenizer_path):
        tokenizer = BPETokenizer.load(config.tokenizer_path)

    if config.coco_zip and os.path.exists(config.coco_zip):
        caps = load_coco_captions(config.coco_zip)
        # 如果没有预训练的 tokenizer，从语料中训练
        if tokenizer is None:
            all_text = " ".join(caps["train"] + caps["val"])
            tokenizer = BPETokenizer(config.model.vocab_size if hasattr(config, "model") else 4096)
            tokenizer.train(all_text, verbose=True)
            if config.tokenizer_path:
                tokenizer.save(config.tokenizer_path)

        # 编码所有 caption，每条加 [BOS]=2 和 [EOS]=3
        all_tokens = []
        for cap in caps["train"]:
            all_tokens.extend([2] + tokenizer.encode(cap) + [3])
        data = torch.tensor(all_tokens, dtype=torch.long)

        # 95% 训练 / 5% 验证
        n = int(0.95 * len(data))
        train_data, val_data = data[:n], data[n:]
        return train_data, val_data, tokenizer

    raise FileNotFoundError(f"未找到 COCO 数据: {config.coco_zip}")


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
