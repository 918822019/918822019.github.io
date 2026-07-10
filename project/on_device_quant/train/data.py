"""
数据加载模块

支持：
  - COCO Captions 数据集加载（小规模，全量载入内存）
  - HuggingFace 流式数据集加载（SkyPile/FineWeb 等，不下载完整数据）
  - BPE tokenizer 训练/加载/编解码
  - PyTorch DataLoader / IterableDataset 封装
"""
import json
import os
import zipfile
import pickle
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

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
        """
        从语料训练 BPE 词表

        special_tokens 占据 id 0-3（<pad>, <bos>, <eos>, <unk>），
        BPE 合并的子词从 id 4 开始。
        """
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            initial_alphabet=ByteLevelPT.alphabet(),
            show_progress=verbose,
        )
        self._tok.train_from_iterator([text], trainer=trainer)
        self._trained = True

    def encode(self, text):
        """
        文本 → token id 列表

        HF tokenizer 的 BPE id 从 4 开始（0-3 是特殊 token），
        下游在每条 caption 前后手动加 [BOS]=2, [EOS]=3 作为序列边界。
        """
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path):
        """从 JSON 文件加载已训练的 tokenizer，跳过 __init__ 避免重复构建"""
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
        # 非重叠切片：每条样本占 seq_len 个 token，步长 = seq_len，无重复
        i = idx * self.seq_len
        x = self.data[i : i + self.seq_len]
        # y 是 x 右移一位：预测下一个 token（语言模型目标）
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
            # 只用前 N 条 caption 训练 BPE（全量 59 万条太慢），1 万条已足够覆盖常用子词
            sample_size = _cfg_get(config, "tokenizer_sample", 10000)
            sample_caps = caps["train"][:sample_size] + caps["val"][:sample_size // 10]
            all_text = " ".join(sample_caps)
            tokenizer = BPETokenizer(_cfg_get(config, "vocab_size", 4096))
            tokenizer.train(all_text, verbose=True)
            if tokenizer_path:
                tokenizer.save(tokenizer_path)

        # 编码所有 caption：每条前后加 [BOS]=2 [EOS]=3 作为序列边界标记
        all_tokens = []
        for cap in caps["train"]:
            all_tokens.extend([2] + tokenizer.encode(cap) + [3])
        data = torch.tensor(all_tokens, dtype=torch.long)

        # 按时间顺序 95/5 划分：前 95% 训练，后 5% 验证
        # COCO captions 无时间序列性，但这样保证验证集是模型没见过的图片描述
        n = int(0.95 * len(data))
        train_data, val_data = data[:n], data[n:]
        return train_data, val_data, tokenizer

    raise FileNotFoundError(f"未找到 COCO 数据: {coco_zip}")


def get_dataloader(token_ids, seq_len, batch_size, num_workers=0, shuffle=True):
    """
    构建 PyTorch DataLoader（内存模式，用于 COCO 等小数据集）

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


# ── 流式数据集（SkyPile/FineWeb 等大数据集）──

class StreamingTextDataset(IterableDataset):
    """
    流式文本数据集：从 HuggingFace datasets 在线读取，不预加载全部数据

    工作原理：
      1. 用 datasets.load_dataset(streaming=True) 打开远程数据集
      2. 逐条取文本 → tokenizer 编码 → 填入 token buffer
      3. buffer 够 seq_len+1 就 yield 一个 (x, y) 样本
      4. 训练时通过持久迭代器持续取数据，不 OOM

    内存占用：仅 buffer 中未消费的 token（通常 < 1MB），不随数据量增长
    """

    def __init__(self, hf_dataset, split, tokenizer, seq_len,
                 text_field="text", max_samples=0, max_size_gb=0,
                 eos_id=3, seed=42):
        """
        Args:
            hf_dataset: HuggingFace dataset name（如 "Skywork/SkyPile-150B"）
            split: 数据集 split（如 "train"）
            tokenizer: 已训练的 BPETokenizer
            seq_len: 每条样本的序列长度
            text_field: 文本字段名（SkyPile 用 "text"，FineWeb 用 "text"）
            max_samples: 最多取多少条文本（0=无限）
            max_size_gb: 最多读取多少 GB 原始文本（0=不限制，按文本字节数累计）
            eos_id: 每段文本之间插入 EOS 作为分隔
            seed: 随机种子（用于 shuffle buffer）
        """
        self.hf_dataset = hf_dataset
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.max_samples = max_samples
        self.max_size_bytes = int(max_size_gb * 1e9) if max_size_gb > 0 else 0
        self.eos_id = eos_id
        self.seed = seed

    def __iter__(self):
        """流式迭代：HF stream → tokenize → 填 buffer → yield (x, y)"""
        from datasets import load_dataset

        # 打开远程数据集流（不下载，逐条读取）
        stream = load_dataset(
            self.hf_dataset, split=self.split, streaming=True
        )
        # shuffle buffer 让数据更随机（非必需，但有助于训练）
        stream = stream.shuffle(seed=self.seed, buffer_size=10_000)

        buffer = []  # token buffer，不够 seq_len+1 就从流中补充
        count = 0
        total_bytes = 0  # 累计读取的原始文本字节数

        for item in stream:
            if self.max_samples and count >= self.max_samples:
                break
            if self.max_size_bytes and total_bytes >= self.max_size_bytes:
                break
            count += 1

            # 取文本 → 编码 → 追加到 buffer
            text = item[self.text_field]
            total_bytes += len(text.encode("utf-8"))
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            buffer.append(self.eos_id)  # 文档边界标记

            # buffer 够切一个样本就 yield
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]  # 消费掉的移出 buffer
                x = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
                y = torch.tensor(chunk[1:self.seq_len + 1], dtype=torch.long)
                yield x, y

        # 流结束后，buffer 里剩余的 token 丢弃（不足一个完整样本）


def get_streaming_dataloader(hf_dataset, split, tokenizer, seq_len, batch_size,
                             text_field="text", max_samples=0, max_size_gb=0, num_workers=0):
    """
    构建流式 DataLoader（用于 SkyPile/FineWeb 等大数据集）

    返回的 DataLoader 是 IterableDataset 模式，
    用 iter(loader) 创建持久迭代器，next(iter) 逐步取 batch。
    """
    ds = StreamingTextDataset(
        hf_dataset=hf_dataset,
        split=split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_field=text_field,
        max_samples=max_samples,
        max_size_gb=max_size_gb,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def train_tokenizer_from_stream(hf_dataset, split, vocab_size, sample_size=10000):
    """
    从流式数据集中采样文本训练 BPE tokenizer

    不下载完整数据集，只取前 sample_size 条文本训练词表。
    """
    from datasets import load_dataset
    stream = load_dataset(hf_dataset, split=split, streaming=True)
    texts = []
    for i, item in enumerate(stream):
        if i >= sample_size:
            break
        texts.append(item["text"])
    all_text = " ".join(texts)
    tokenizer = BPETokenizer(vocab_size)
    tokenizer.train(all_text, verbose=True)
    return tokenizer


def load_streaming_data(config):
    """
    流式数据加载（SkyPile/FineWeb 等）

    流程：
      1. 加载已有 tokenizer，或从数据流中采样训练
      2. 构建 train/val 两个流式 DataLoader
      3. 返回 (train_loader, val_loader, tokenizer)

    与 load_data() 的区别：不把全部数据载入内存，
    而是用 IterableDataset 持续从远程流中读取。
    """
    tokenizer_path = _cfg_get(config, "tokenizer_path", "")
    hf_dataset = _cfg_get(config, "hf_dataset", "Skywork/SkyPile-150B")
    hf_train_split = _cfg_get(config, "hf_train_split", "train")
    hf_val_split = _cfg_get(config, "hf_val_split", "")
    seq_len = _cfg_get(config, "seq_len", 512)
    batch_size = _cfg_get(config, "batch_size", 8)
    vocab_size = _cfg_get(config, "vocab_size", 8192)
    max_samples = _cfg_get(config, "max_samples", 0)
    max_size_gb = _cfg_get(config, "max_size_gb", 0)
    text_field = _cfg_get(config, "text_field", "text")

    # 加载或训练 tokenizer
    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer is None:
        print(f"从 {hf_dataset} 采样训练 BPE tokenizer (vocab={vocab_size})...")
        sample_size = _cfg_get(config, "tokenizer_sample", 10000)
        tokenizer = train_tokenizer_from_stream(
            hf_dataset, hf_train_split, vocab_size, sample_size
        )
        if tokenizer_path:
            tokenizer.save(tokenizer_path)

    # 构建训练集 DataLoader
    train_loader = get_streaming_dataloader(
        hf_dataset, hf_train_split, tokenizer, seq_len, batch_size,
        text_field=text_field, max_samples=max_samples, max_size_gb=max_size_gb,
    )

    # 构建验证集 DataLoader
    # 如果没有独立的 val split，从 train 流中取少量样本作为验证
    if hf_val_split:
        val_max = min(max_samples, 1000) if max_samples else 1000
        val_loader = get_streaming_dataloader(
            hf_dataset, hf_val_split, tokenizer, seq_len, batch_size,
            text_field=text_field, max_samples=val_max, max_size_gb=0,
        )
    else:
        # 没有 val split：从 train 流中取少量样本（模拟验证集）
        val_max = 1000
        val_loader = get_streaming_dataloader(
            hf_dataset, hf_train_split, tokenizer, seq_len, batch_size,
            text_field=text_field, max_samples=val_max, max_size_gb=0,
        )

    return train_loader, val_loader, tokenizer
