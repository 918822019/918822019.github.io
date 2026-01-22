"""
Embedding模型对比学习训练

支持模型:
- Qwen3-Embedding-0.6B
- BERT (bert-base-chinese)
- BGE (BAAI/bge-base-zh-v1.5)
- M3E, Text2Vec等

使用对比学习(Contrastive Learning)方法进行微调训练
"""

import os
import random
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from tqdm import tqdm


# ==================== 配置 ====================

MODEL_CONFIGS = {
    'bert-base-chinese': {
        'model_name': 'bert-base-chinese',
        'pooling': 'cls',
        'description': 'BERT中文基础模型'
    },
    'bge-base-zh': {
        'model_name': 'BAAI/bge-base-zh-v1.5',
        'pooling': 'cls',
        'description': 'BGE中文Embedding模型'
    },
    'bge-small-zh': {
        'model_name': 'BAAI/bge-small-zh-v1.5',
        'pooling': 'cls',
        'description': 'BGE中文小型Embedding模型'
    },
    'qwen3-embedding': {
        'model_name': 'Qwen/Qwen3-Embedding-0.6B',
        'pooling': 'last',
        'description': 'Qwen3 Embedding模型 0.6B'
    },
    'text2vec-chinese': {
        'model_name': 'shibing624/text2vec-base-chinese',
        'pooling': 'mean',
        'description': 'Text2Vec中文模型'
    },
    'm3e-base': {
        'model_name': 'moka-ai/m3e-base',
        'pooling': 'mean',
        'description': 'M3E中文Embedding模型'
    }
}

# Mock金融实体数据
MOCK_ENTITIES = {
    "公司": [
        ("阿里巴巴集团", "Alibaba Group", "电商、云计算"),
        ("腾讯控股", "Tencent Holdings", "社交、游戏"),
        ("京东集团", "JD.com", "电商、物流"),
        ("百度公司", "Baidu Inc", "搜索、AI"),
        ("美团点评", "Meituan", "本地生活服务"),
        ("拼多多", "Pinduoduo", "电商、社交"),
        ("字节跳动", "ByteDance", "短视频、信息流"),
        ("小米集团", "Xiaomi Corp", "智能硬件、IoT"),
        ("华为技术", "Huawei Technologies", "通信设备、手机"),
        ("中国平安", "Ping An Insurance", "保险、金融科技"),
    ],
    "基金": [
        ("易方达蓝筹精选", "E Fund Blue Chip", "股票型基金"),
        ("华夏成长混合", "China Growth Mixed", "混合型基金"),
        ("南方消费升级", "Southern Consumer", "消费主题基金"),
        ("招商中证白酒", "CMF Baijiu Index", "行业指数基金"),
        ("广发科技先锋", "GF Tech Pioneer", "科技主题基金"),
        ("富国天惠成长", "Fullgoal Tianhui", "成长型基金"),
        ("兴全合润混合", "Xingquan Herun", "混合型基金"),
        ("景顺长城新兴", "IGWFMC Emerging", "新兴产业基金"),
        ("嘉实新能源", "Harvest New Energy", "新能源主题基金"),
        ("汇添富消费", "HTFF Consumer", "消费主题基金"),
    ],
    "指标": [
        ("市盈率", "P/E Ratio", "估值指标"),
        ("市净率", "P/B Ratio", "估值指标"),
        ("净资产收益率", "ROE", "盈利能力指标"),
        ("资产负债率", "Debt Ratio", "偿债能力指标"),
        ("毛利率", "Gross Margin", "盈利能力指标"),
        ("净利润增长率", "Net Profit Growth", "成长性指标"),
        ("营收增长率", "Revenue Growth", "成长性指标"),
        ("股息率", "Dividend Yield", "分红指标"),
        ("换手率", "Turnover Rate", "流动性指标"),
        ("夏普比率", "Sharpe Ratio", "风险调整收益"),
    ]
}


# ==================== 工具函数 ====================

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ==================== 数据生成 ====================

def generate_contrastive_pairs(entities_dict: Dict, num_pairs: int = 1000) -> List[Dict]:
    """
    生成对比学习的数据对
    正样本：同一实体的不同表述（中文名、英文名、描述）
    负样本：不同实体的表述
    """
    pairs = []
    all_entities = []

    for category, entity_list in entities_dict.items():
        for entity in entity_list:
            all_entities.append((category, entity))

    for _ in range(num_pairs):
        anchor_idx = random.randint(0, len(all_entities) - 1)
        anchor_category, anchor_entity = all_entities[anchor_idx]

        anchor_texts = [anchor_entity[0], anchor_entity[1], f"{anchor_entity[0]}是{anchor_entity[2]}"]
        anchor_text = random.choice(anchor_texts)

        positive_texts = [t for t in anchor_texts if t != anchor_text]
        positive_text = random.choice(positive_texts) if positive_texts else anchor_texts[0]

        neg_idx = random.randint(0, len(all_entities) - 1)
        while neg_idx == anchor_idx:
            neg_idx = random.randint(0, len(all_entities) - 1)
        neg_category, neg_entity = all_entities[neg_idx]
        neg_texts = [neg_entity[0], neg_entity[1], f"{neg_entity[0]}是{neg_entity[2]}"]
        negative_text = random.choice(neg_texts)

        pairs.append({
            'anchor': anchor_text,
            'positive': positive_text,
            'negative': negative_text,
            'anchor_category': anchor_category,
            'negative_category': neg_category
        })

    return pairs


# ==================== 数据集 ====================

class ContrastiveDataset(Dataset):
    """对比学习数据集"""

    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        anchor_encoding = self.tokenizer(
            pair['anchor'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        positive_encoding = self.tokenizer(
            pair['positive'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        negative_encoding = self.tokenizer(
            pair['negative'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0),
        }


# ==================== 模型 ====================

class EmbeddingModel(nn.Module):
    """通用Embedding模型包装器"""

    def __init__(self, model_name: str, pooling_strategy: str = 'cls'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.pooling_strategy = pooling_strategy
        self.hidden_size = self.encoder.config.hidden_size

    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.pooling_strategy == 'cls':
            embeddings = hidden_states[:, 0, :]
        elif self.pooling_strategy == 'mean':
            embeddings = self.mean_pooling(hidden_states, attention_mask)
        elif self.pooling_strategy == 'last':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        else:
            embeddings = hidden_states[:, 0, :]

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @torch.no_grad()
    def get_embedding(self, text: str, tokenizer, max_length: int = 128) -> np.ndarray:
        """获取单个文本的embedding"""
        self.eval()
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoding['attention_mask'].to(next(self.parameters()).device)
        embedding = self(input_ids, attention_mask)
        return embedding.cpu().numpy()


# ==================== 损失函数 ====================

class TripletLoss(nn.Module):
    """Triplet Loss with margin"""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class InfoNCELoss(nn.Module):
    """InfoNCE Loss"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative) / self.temperature

        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """Multiple Negatives Ranking Loss (in-batch negatives)"""

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor = None) -> torch.Tensor:
        scores = torch.mm(anchor, positive.t()) * self.scale
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)
        return loss


# ==================== 训练器 ====================

class ContrastiveTrainer:
    """对比学习训练器"""

    def __init__(
        self,
        model: EmbeddingModel,
        tokenizer,
        train_pairs: List[Dict],
        val_pairs: List[Dict],
        loss_fn: str = 'triplet',
        batch_size: int = 16,
        lr: float = 2e-5,
        epochs: int = 3,
        device: torch.device = None
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_dataset = ContrastiveDataset(train_pairs, tokenizer)
        self.val_dataset = ContrastiveDataset(val_pairs, tokenizer)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # 损失函数
        if loss_fn == 'triplet':
            self.criterion = TripletLoss(margin=0.5)
        elif loss_fn == 'infonce':
            self.criterion = InfoNCELoss(temperature=0.07)
        elif loss_fn == 'mnrl':
            self.criterion = MultipleNegativesRankingLoss(scale=20.0)
        else:
            self.criterion = TripletLoss(margin=0.5)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        total_steps = len(self.train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            anchor_emb = self.model(
                batch['anchor_input_ids'].to(self.device),
                batch['anchor_attention_mask'].to(self.device)
            )
            positive_emb = self.model(
                batch['positive_input_ids'].to(self.device),
                batch['positive_attention_mask'].to(self.device)
            )
            negative_emb = self.model(
                batch['negative_input_ids'].to(self.device),
                batch['negative_attention_mask'].to(self.device)
            )

            loss = self.criterion(anchor_emb, positive_emb, negative_emb)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            anchor_emb = self.model(
                batch['anchor_input_ids'].to(self.device),
                batch['anchor_attention_mask'].to(self.device)
            )
            positive_emb = self.model(
                batch['positive_input_ids'].to(self.device),
                batch['positive_attention_mask'].to(self.device)
            )
            negative_emb = self.model(
                batch['negative_input_ids'].to(self.device),
                batch['negative_attention_mask'].to(self.device)
            )

            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self) -> Tuple[List[float], List[float]]:
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*50}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("✓ New best validation loss!")

        return self.train_losses, self.val_losses


# ==================== 评估 ====================

def evaluate_embedding_quality(
    model: EmbeddingModel,
    tokenizer,
    test_pairs: List[Dict]
) -> Dict[str, float]:
    """评估embedding质量"""
    model.eval()

    correct_rankings = 0
    pos_similarities = []
    neg_similarities = []

    for pair in tqdm(test_pairs, desc='Evaluating'):
        anchor_emb = model.get_embedding(pair['anchor'], tokenizer)
        pos_emb = model.get_embedding(pair['positive'], tokenizer)
        neg_emb = model.get_embedding(pair['negative'], tokenizer)

        pos_sim = cosine_similarity(anchor_emb, pos_emb)[0][0]
        neg_sim = cosine_similarity(anchor_emb, neg_emb)[0][0]

        pos_similarities.append(pos_sim)
        neg_similarities.append(neg_sim)

        if pos_sim > neg_sim:
            correct_rankings += 1

    accuracy = correct_rankings / len(test_pairs)
    avg_pos_sim = np.mean(pos_similarities)
    avg_neg_sim = np.mean(neg_similarities)
    margin = avg_pos_sim - avg_neg_sim

    return {
        'accuracy': accuracy,
        'avg_positive_similarity': avg_pos_sim,
        'avg_negative_similarity': avg_neg_sim,
        'margin': margin
    }


# ==================== 可视化 ====================

def plot_training_curves(results: Dict, save_path: str = 'training_curves.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for model_key, result in results.items():
        epochs = range(1, len(result['train_losses']) + 1)
        ax1.plot(epochs, result['train_losses'], marker='o', label=model_key)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for model_key, result in results.items():
        epochs = range(1, len(result['val_losses']) + 1)
        ax2.plot(epochs, result['val_losses'], marker='s', label=model_key)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def visualize_embeddings(
    model: EmbeddingModel,
    tokenizer,
    entities_dict: Dict,
    save_path: str = 'embedding_visualization.png'
):
    """使用t-SNE可视化embeddings"""
    model.eval()

    embeddings = []
    labels = []
    texts = []

    for category, entity_list in entities_dict.items():
        for entity in entity_list:
            text = entity[0]
            emb = model.get_embedding(text, tokenizer)
            embeddings.append(emb.squeeze())
            labels.append(category)
            texts.append(text)

    embeddings = np.array(embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))

    categories = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    for cat, color in zip(categories, colors):
        mask = [l == cat for l in labels]
        points = embeddings_2d[mask]
        plt.scatter(points[:, 0], points[:, 1], c=[color], label=cat, s=100, alpha=0.7)

        for i, (x, y) in enumerate(points):
            text_idx = [j for j, m in enumerate(mask) if m][i]
            plt.annotate(texts[text_idx], (x, y), fontsize=8, alpha=0.8)

    plt.legend()
    plt.title('Embedding Space Visualization (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Embedding可视化已保存到: {save_path}")
    plt.close()


# ==================== 模型保存/加载 ====================

def save_model(model: EmbeddingModel, tokenizer, model_key: str, save_dir: str = './saved_models'):
    """保存训练好的模型"""
    model_save_path = os.path.join(save_dir, model_key)
    os.makedirs(model_save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pt'))
    tokenizer.save_pretrained(model_save_path)

    print(f"模型已保存到: {model_save_path}")


# ==================== 推理 ====================

def similarity_search(
    query: str,
    corpus: List[str],
    model: EmbeddingModel,
    tokenizer,
    top_k: int = 5
) -> List[Dict]:
    """相似度检索"""
    model.eval()

    query_emb = model.get_embedding(query, tokenizer)

    corpus_embs = []
    for text in corpus:
        emb = model.get_embedding(text, tokenizer)
        corpus_embs.append(emb.squeeze())
    corpus_embs = np.array(corpus_embs)

    similarities = cosine_similarity(query_emb, corpus_embs)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'text': corpus[idx],
            'similarity': float(similarities[idx])
        })

    return results


# ==================== 主函数 ====================

def train_model(
    model_key: str,
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    loss_fn: str = 'triplet',
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: torch.device = None,
    model_path: str = None,
    pooling: str = 'last'
) -> Dict:
    """训练单个模型"""
    # 处理本地模型
    if model_key == 'local':
        if model_path is None:
            raise ValueError("使用local模型时必须指定--model_path")
        config = {
            'model_name': model_path,
            'pooling': pooling,
            'description': f'本地模型: {model_path}'
        }
    else:
        config = MODEL_CONFIGS[model_key]

    print(f"\n{'#'*60}")
    print(f"训练模型: {model_key}")
    print(f"模型名称: {config['model_name']}")
    print(f"Pooling策略: {config['pooling']}")
    print(f"损失函数: {loss_fn}")
    print(f"{'#'*60}")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    model = EmbeddingModel(config['model_name'], pooling_strategy=config['pooling'])

    trainer = ContrastiveTrainer(
        model=model,
        tokenizer=tokenizer,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        loss_fn=loss_fn,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        device=device
    )

    train_losses, val_losses = trainer.train()

    return {
        'model': model,
        'tokenizer': tokenizer,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='Embedding模型对比学习训练')
    parser.add_argument('--model', type=str, default='bge-small-zh',
                        choices=list(MODEL_CONFIGS.keys()) + ['local'],
                        help='选择要训练的模型，使用local时需配合--model_path')
    parser.add_argument('--model_path', type=str, default=None,
                        help='本地模型路径（当--model=local时使用）')
    parser.add_argument('--pooling', type=str, default='last',
                        choices=['cls', 'mean', 'last'],
                        help='Pooling策略（当使用本地模型时）')
    parser.add_argument('--loss', type=str, default='triplet',
                        choices=['triplet', 'infonce', 'mnrl'],
                        help='损失函数类型')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--train_pairs', type=int, default=800, help='训练样本数')
    parser.add_argument('--val_pairs', type=int, default=200, help='验证样本数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='模型保存目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化embedding')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device()
    print(f"Using device: {device}")

    # 生成训练数据
    print("\n生成训练数据...")
    train_pairs = generate_contrastive_pairs(MOCK_ENTITIES, num_pairs=args.train_pairs)
    val_pairs = generate_contrastive_pairs(MOCK_ENTITIES, num_pairs=args.val_pairs)
    print(f"训练集大小: {len(train_pairs)}")
    print(f"验证集大小: {len(val_pairs)}")

    # 训练模型
    result = train_model(
        model_key=args.model,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        loss_fn=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        model_path=args.model_path,
        pooling=args.pooling
    )

    # 评估模型
    print("\n评估模型...")
    eval_result = evaluate_embedding_quality(
        result['model'],
        result['tokenizer'],
        val_pairs
    )
    print(f"准确率: {eval_result['accuracy']:.4f}")
    print(f"平均正样本相似度: {eval_result['avg_positive_similarity']:.4f}")
    print(f"平均负样本相似度: {eval_result['avg_negative_similarity']:.4f}")
    print(f"Margin: {eval_result['margin']:.4f}")

    # 保存模型
    save_model(result['model'], result['tokenizer'], args.model, args.save_dir)

    # 可视化
    if args.visualize:
        visualize_embeddings(
            result['model'],
            result['tokenizer'],
            MOCK_ENTITIES,
            save_path='embedding_visualization.png'
        )

    # 测试相似度检索
    print("\n测试相似度检索...")
    corpus = []
    for category, entities in MOCK_ENTITIES.items():
        for entity in entities:
            corpus.append(entity[0])
            corpus.append(entity[1])

    test_queries = ["阿里巴巴", "P/E Ratio", "消费基金"]
    for query in test_queries:
        print(f"\n查询: '{query}'")
        print("-" * 40)
        results = similarity_search(query, corpus, result['model'], result['tokenizer'], top_k=5)
        for i, res in enumerate(results, 1):
            print(f"  {i}. {res['text']} (相似度: {res['similarity']:.4f})")

    print("\n训练完成!")


if __name__ == '__main__':
    main()
