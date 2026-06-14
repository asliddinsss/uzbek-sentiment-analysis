"""
evaluate.py
-----------
Metrics, confusion matrix, domain breakdown, training curves.
"""

import os
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

LABEL_NAMES   = ["Negative", "Neutral", "Positive"]
LABEL_COLORS  = ["#ef5350", "#ffd54f", "#66bb6a"]
DOMAIN_COLORS = {"news": "#4fc3f7", "telegram": "#ef5350", "review": "#66bb6a"}
BG, PANEL, BORDER = "#0f1117", "#1a1d27", "#333344"


def dark_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_edgecolor(BORDER)
    ax.tick_params(colors="#aaaaaa")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.grid(alpha=0.13, color="white")


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_domain_metrics(y_true, y_pred, domains) -> Dict[str, Dict]:
    buckets = defaultdict(lambda: {"true": [], "pred": []})
    for t, p, d in zip(y_true, y_pred, domains):
        buckets[d]["true"].append(t)
        buckets[d]["pred"].append(p)
    return {d: compute_metrics(v["true"], v["pred"]) for d, v in buckets.items()}


# ─────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred,
                          save_path="results/confusion_matrix.png"):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)

    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".1%"], ["Counts", "Normalized (row %)"]
    ):
        ax.set_facecolor(PANEL)
        sns.heatmap(data, annot=True, fmt=fmt, ax=ax, cmap="RdYlGn",
                    linewidths=1.5, linecolor=BG,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                    annot_kws={"size":14,"weight":"bold","color":"white"},
                    cbar_kws={"shrink":0.8})
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", color="#aaaaaa", fontsize=11)
        ax.set_ylabel("Actual",    color="#aaaaaa", fontsize=11)
        ax.tick_params(colors="#cccccc", labelsize=10)
        ax.yaxis.set_tick_params(rotation=0)
        ax.collections[0].colorbar.ax.tick_params(colors="#aaaaaa")

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fig.suptitle(f"Confusion Matrix | Accuracy: {acc:.1%} | Macro F1: {f1:.3f}",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Plot] Confusion matrix → {save_path}")


# ─────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────

def plot_training_curves(log_csv="results/training_log.csv",
                         save_path="results/training_curves.png"):
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)

    for ax, metric, title in zip(axes, ["loss","acc","f1"],
                                  ["Loss","Accuracy","Macro F1"]):
        dark_ax(ax)
        ax.plot(df["epoch"], df[f"train_{metric}"],
                color="#4fc3f7", lw=2.5, label="Train", marker="o", ms=5)
        ax.plot(df["epoch"], df[f"val_{metric}"],
                color="#ef5350", lw=2.5, label="Val", ls="--", marker="s", ms=5)

        # Phase line
        if "phase" in df.columns:
            switch = df[df["phase"]=="full"]["epoch"].min()
            if not np.isnan(switch):
                ax.axvline(x=switch-0.5, color="#ffd54f", ls=":", lw=1.5,
                           label="Unfreeze")

        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.legend(fontsize=9, facecolor=PANEL, labelcolor="white", edgecolor=BORDER)

    fig.suptitle("XLM-RoBERTa — Uzbek Sentiment Analysis Training",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Plot] Training curves → {save_path}")


# ─────────────────────────────────────────────
# Domain breakdown
# ─────────────────────────────────────────────

def plot_domain_breakdown(domain_metrics: Dict,
                          save_path="results/domain_breakdown.png"):
    domains = list(domain_metrics.keys())
    accs    = [domain_metrics[d]["accuracy"] for d in domains]
    f1s     = [domain_metrics[d]["f1"]       for d in domains]
    colors  = [DOMAIN_COLORS.get(d, "#aaaaaa") for d in domains]

    x = np.arange(len(domains))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)
    dark_ax(ax)

    b1 = ax.bar(x - w/2, accs, w, color=colors, alpha=0.92, edgecolor=BG, label="Accuracy")
    b2 = ax.bar(x + w/2, f1s,  w, color=colors, alpha=0.50, edgecolor=BG, label="Macro F1")

    for bar, val in zip(list(b1)+list(b2), accs+f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains],
                       color="white", fontsize=13, fontweight="bold")
    ax.set_ylim(0.72, 0.98)
    ax.set_ylabel("Score", color="#aaaaaa")
    ax.set_title("Per-Domain Performance — Uzbek Sentiment Analysis\n"
                 "XLM-RoBERTa (trained on mixed domain)",
                 color="white", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10, facecolor=PANEL, labelcolor="white", edgecolor=BORDER)

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Plot] Domain breakdown → {save_path}")


# ─────────────────────────────────────────────
# Full test evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, test_loader, device, results_dir="results"):
    import torch
    model.eval()
    all_preds, all_labels, all_domains = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["label"].to(device)
            preds = model(ids, mask).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbl.cpu().tolist())
            all_domains.extend(batch["domain"])

    metrics       = compute_metrics(all_labels, all_preds)
    domain_metrics = per_domain_metrics(all_labels, all_preds, all_domains)

    print("\n" + "="*55)
    print("  TEST SET RESULTS")
    print("="*55)
    for k, v in metrics.items():
        print(f"  {k.capitalize():>10}: {v:.4f}")
    print("\nPer-domain:")
    for d, dm in domain_metrics.items():
        print(f"  [{d}] Acc: {dm['accuracy']:.4f} | F1: {dm['f1']:.4f}")
    print("\nDetailed report:")
    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))

    plot_confusion_matrix(all_labels, all_preds,
                          os.path.join(results_dir, "confusion_matrix.png"))
    plot_domain_breakdown(domain_metrics,
                          os.path.join(results_dir, "domain_breakdown.png"))

    return metrics, domain_metrics
