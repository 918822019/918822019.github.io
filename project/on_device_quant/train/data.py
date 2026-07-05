import json, os, zipfile, pickle
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader


class BPETokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}

    def train(self, text, verbose=False):
        tokens = list(text.encode("utf-8"))
        for i in range(self.vocab_size - 256):
            counts = Counter()
            for a, b in zip(tokens, tokens[1:]):
                counts[(a, b)] += 1
            if not counts:
                break
            best = max(counts, key=counts.get)
            new_id = 256 + i
            self.merges.append((best[0], best[1], new_id))
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
        self.vocab = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in self.merges:
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
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
        return b"".join(self.vocab.get(i, b"?") for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"merges": self.merges, "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        tok = cls(d["vocab_size"])
        tok.merges = d["merges"]
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in tok.merges:
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        return tok


class CaptionDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.n = len(self.data) - seq_len - 1

    def __len__(self):
        return self.n // self.seq_len

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = self.data[i : i + self.seq_len]
        y = self.data[i + 1 : i + self.seq_len + 1]
        return x, y


class SequenceDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.n = len(self.data) - seq_len - 1

    def __len__(self):
        return max(0, self.n // self.seq_len)

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = self.data[i : i + self.seq_len]
        y = self.data[i + 1 : i + self.seq_len + 1]
        return x, y


def load_coco_captions(zip_path):
    caps = {"train": [], "val": []}
    with zipfile.ZipFile(zip_path) as z:
        for split in ["train", "val"]:
            with z.open(f"annotations/captions_{split}2017.json") as f:
                data = json.load(f)
            caps[split] = [a["caption"].lower() for a in data["annotations"]]
    return caps


def build_multimodal_sequences(caps_dict, min_caps=2, max_per_image=5):
    img_to_caps = defaultdict(list)
    sep_chars = [".", "!", "?"]

    with zipfile.ZipFile("") as z:
        pass

    for split in ["train", "val"]:
        pass

    all_seqs = []
    for split in ["train", "val"]:
        if not caps_dict[split]:
            continue
        img_caps = defaultdict(list)
        for cap in caps_dict[split]:
            img_caps[cap[:10]].append(cap)
        for img_id, group_caps in img_caps.items():
            if len(group_caps) >= min_caps:
                selected = group_caps[:max_per_image]
                sep = sep_chars[hash(img_id) % len(sep_chars)]
                seq = f" {sep} ".join(selected)
                all_seqs.append(seq)
    return all_seqs


def load_data(config):
    tokenizer = None
    if config.tokenizer_path and os.path.exists(config.tokenizer_path):
        tokenizer = BPETokenizer.load(config.tokenizer_path)

    if config.coco_zip and os.path.exists(config.coco_zip):
        caps = load_coco_captions(config.coco_zip)
        if tokenizer is None:
            all_text = " ".join(caps["train"] + caps["val"])
            tokenizer = BPETokenizer(config.model.vocab_size if hasattr(config, "model") else 4096)
            tokenizer.train(all_text, verbose=True)
            if config.tokenizer_path:
                tokenizer.save(config.tokenizer_path)

        all_tokens = []
        for cap in caps["train"]:
            all_tokens.extend([2] + tokenizer.encode(cap) + [3])
        data = torch.tensor(all_tokens, dtype=torch.long)
        n = int(0.95 * len(data))
        train_data, val_data = data[:n], data[n:]
        return train_data, val_data, tokenizer

    raise FileNotFoundError(f"Data not found: {config.coco_zip}")


def get_dataloader(token_ids, seq_len, batch_size, num_workers=0, shuffle=True):
    ds = SequenceDataset(token_ids, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)
