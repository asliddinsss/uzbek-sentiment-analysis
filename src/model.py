"""
model.py
--------
Three model tiers for Uzbek sentiment analysis:

1. LexiconModel   — rule-based, zero training, Uzbek word list
2. BaselineModel  — TF-IDF + Logistic Regression
3. TransformerModel — XLM-RoBERTa / mBERT fine-tuned
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

from dataset import LABEL2ID, ID2LABEL, POSITIVE_WORDS, NEGATIVE_WORDS, lexicon_predict

NUM_CLASSES = 3


# ─────────────────────────────────────────────
# 1. Lexicon (rule-based)
# ─────────────────────────────────────────────

class LexiconModel:
    """
    Zero-shot rule-based baseline using an Uzbek sentiment lexicon.
    No training required — useful as a sanity-check lower bound.
    """

    def predict(self, texts):
        return [lexicon_predict(t) for t in texts]

    def predict_proba(self, texts):
        # Soft scores based on word count ratios
        probs = []
        for text in texts:
            words     = text.lower().split()
            pos_score = sum(1 for w in words if w in POSITIVE_WORDS)
            neg_score = sum(1 for w in words if w in NEGATIVE_WORDS)
            total     = pos_score + neg_score + 1e-6
            neutral_p = max(0.0, 1.0 - (pos_score + neg_score) / max(len(words), 1))
            raw = np.array([neg_score / total, neutral_p, pos_score / total])
            probs.append(raw / raw.sum())
        return np.array(probs)


# ─────────────────────────────────────────────
# 2. TF-IDF + Logistic Regression
# ─────────────────────────────────────────────

class BaselineModel:
    """
    Classical ML baseline: TF-IDF + Logistic Regression.
    Works at character n-gram level for better Uzbek morphology handling.
    """

    def __init__(self, max_features: int = 30000):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),
                analyzer="char_wb",         # Character n-grams handle Uzbek morphology better
                sublinear_tf=True,
                min_df=2,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                multi_class="multinomial",
                solver="lbfgs",
            )),
        ])

    def fit(self, texts, labels):
        print("[Baseline] Training TF-IDF (char n-gram) + Logistic Regression...")
        self.pipeline.fit(texts, labels)
        print("[Baseline] Done.")
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

    def save(self, path: str = "models/baseline.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"[Baseline] Saved → {path}")

    def load(self, path: str = "models/baseline.pkl"):
        self.pipeline = joblib.load(path)
        return self


# ─────────────────────────────────────────────
# 3. Transformer (XLM-RoBERTa / mBERT)
# ─────────────────────────────────────────────

class TransformerModel(nn.Module):
    """
    Fine-tuned multilingual transformer for Uzbek sentiment.

    Architecture:
        XLM-RoBERTa-base → [CLS] → Dropout → FC(768→256) → GELU → FC(256→3)

    Args:
        model_name:  HuggingFace identifier
        num_classes: 3 (Negative / Neutral / Positive)
        dropout:     Dropout on [CLS] embedding
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name
        self.config     = AutoConfig.from_pretrained(model_name)
        self.backbone   = AutoModel.from_pretrained(model_name)
        hidden          = self.config.hidden_size   # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(256, num_classes),
        )

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] {model_name} | Total: {total:,} | Trainable: {trainable:,}")

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        print("[Model] Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True
        print("[Model] Backbone unfrozen.")

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]   # [CLS] token
        logits = self.classifier(cls)
        return logits

    def save_pretrained(self, save_dir: str = "models/best_model"):
        os.makedirs(save_dir, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        torch.save(self.classifier.state_dict(),
                   os.path.join(save_dir, "classifier_head.pt"))
        print(f"[Model] Saved → {save_dir}/")

    def load_pretrained(self, save_dir: str, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModel.from_pretrained(save_dir).to(device)
        self.classifier.load_state_dict(
            torch.load(os.path.join(save_dir, "classifier_head.pt"), map_location=device)
        )
        return self


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_model(model_type: str = "xlmroberta", **kwargs):
    if model_type == "lexicon":
        return LexiconModel()
    if model_type == "baseline":
        return BaselineModel(**kwargs)
    model_map = {
        "xlmroberta": "xlm-roberta-base",
        "mbert":      "bert-base-multilingual-cased",
    }
    assert model_type in model_map, f"Unknown model: {model_type}"
    return TransformerModel(model_name=model_map[model_type], **kwargs)


if __name__ == "__main__":
    model = build_model("xlmroberta")
    ids   = torch.randint(0, 250002, (2, 64))
    mask  = torch.ones(2, 64, dtype=torch.long)
    print("Output:", model(ids, mask).shape)   # (2, 3)
