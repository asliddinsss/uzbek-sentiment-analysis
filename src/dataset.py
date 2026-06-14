"""
dataset.py
----------
Domain-aware dataset class for Uzbek sentiment analysis.

Domains: news | telegram | review
Labels:  0=Negative | 1=Neutral | 2=Positive

CSV format expected:
    text    : Uzbek text string
    label   : negative | neutral | positive
    domain  : news | telegram | review
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LABEL2ID  = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
LABEL_EMOJI = {0: "😠", 1: "😐", 2: "😊"}
DOMAINS   = {"news", "telegram", "review"}

DOMAIN_COLORS = {
    "news":     "#4fc3f7",
    "telegram": "#ef5350",
    "review":   "#66bb6a",
}


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def clean_uzbek_text(text: str) -> str:
    """
    Basic cleaning for Uzbek text.
    Handles both Latin (post-1995) and Cyrillic (legacy) scripts.
    """
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())   # Normalize whitespace
    text = text.strip()
    # Remove URLs
    import re
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove excessive punctuation repetition (e.g. "!!!!!!")
    text = re.sub(r"([!?.]){3,}", r"\1\1", text)
    return text


# ─────────────────────────────────────────────
# Simple Uzbek Sentiment Lexicon (rule-based baseline)
# ─────────────────────────────────────────────

POSITIVE_WORDS = {
    "yaxshi", "ajoyib", "zo'r", "mukammal", "mamnun", "baxtli",
    "yoqimli", "chiroyli", "foydali", "rahmat", "tabriklayman",
    "yutuq", "muvaffaqiyat", "tavsiya", "sevaman", "go'zal",
}

NEGATIVE_WORDS = {
    "yomon", "dahshatli", "xato", "muammo", "afsuski", "noto'g'ri",
    "qiyin", "og'ir", "xafa", "norozi", "kamchilik", "shikoyat",
    "ishlamadi", "buzilgan", "kechikdi", "umidsiz",
}


def lexicon_predict(text: str) -> int:
    """Rule-based sentiment using Uzbek lexicon."""
    words = set(text.lower().split())
    pos_score = len(words & POSITIVE_WORDS)
    neg_score = len(words & NEGATIVE_WORDS)
    if pos_score > neg_score:
        return 2   # Positive
    elif neg_score > pos_score:
        return 0   # Negative
    return 1       # Neutral


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────

class UzbekSentimentDataset(Dataset):
    """
    PyTorch Dataset for Uzbek sentiment classification.

    Args:
        csv_path:       Path to CSV with columns: text, label, domain
        tokenizer_name: HuggingFace model identifier
        max_length:     Max token sequence length
        domain_filter:  If set, only load one domain
        augment:        Token dropout augmentation
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "xlm-roberta-base",
        max_length: int = 128,
        domain_filter: Optional[str] = None,
        augment: bool = False,
    ):
        self.max_length = max_length
        self.augment    = augment

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        df = pd.read_csv(csv_path)
        df = self._preprocess(df, domain_filter)

        self.texts   = df["text"].tolist()
        self.labels  = df["label_id"].tolist()
        self.domains = df["domain"].tolist()

        label_dist = {ID2LABEL[i]: self.labels.count(i) for i in range(3)}
        print(f"[Dataset] {len(self.texts)} samples | {label_dist}")
        if domain_filter:
            print(f"  Domain filter: {domain_filter}")

    def _preprocess(self, df: pd.DataFrame, domain_filter: Optional[str]) -> pd.DataFrame:
        df["text"]  = df["text"].apply(clean_uzbek_text)
        df = df[df["text"].str.len() > 10]
        df["label"] = df["label"].str.lower().str.strip()
        df = df[df["label"].isin(LABEL2ID)]
        df["label_id"] = df["label"].map(LABEL2ID)

        if "domain" not in df.columns:
            df["domain"] = "unknown"
        if domain_filter:
            df = df[df["domain"] == domain_filter]

        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text  = self.texts[idx]
        label = self.labels[idx]

        # Token dropout augmentation
        if self.augment and np.random.rand() < 0.15:
            words = text.split()
            keep  = np.random.rand(len(words)) > 0.10
            text  = " ".join([w for w, k in zip(words, keep) if k]) or text

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
            "domain":         self.domains[idx],
        }

    def get_class_weights(self) -> torch.Tensor:
        counts  = np.bincount(self.labels, minlength=3).astype(float)
        weights = 1.0 / (counts + 1e-6)
        return torch.tensor(weights / weights.sum() * 3, dtype=torch.float)

    def get_sampler(self) -> WeightedRandomSampler:
        cw = self.get_class_weights().numpy()
        sw = [cw[l] for l in self.labels]
        return WeightedRandomSampler(sw, num_samples=len(self.labels), replacement=True)


# ─────────────────────────────────────────────
# Collate + factory
# ─────────────────────────────────────────────

def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label":          torch.stack([b["label"]          for b in batch]),
        "domain":         [b["domain"] for b in batch],
    }


def get_dataloaders(
    data_dir: str = "data/processed",
    tokenizer_name: str = "xlm-roberta-base",
    max_length: int = 128,
    batch_size: int = 16,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = UzbekSentimentDataset(
        os.path.join(data_dir, "train.csv"),
        tokenizer_name=tokenizer_name,
        max_length=max_length, augment=True,
    )
    val_ds = UzbekSentimentDataset(
        os.path.join(data_dir, "val.csv"),
        tokenizer_name=tokenizer_name,
        max_length=max_length, augment=False,
    )
    test_ds = UzbekSentimentDataset(
        os.path.join(data_dir, "test.csv"),
        tokenizer_name=tokenizer_name,
        max_length=max_length, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_ds.get_sampler(),
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    return train_loader, val_loader, test_loader
