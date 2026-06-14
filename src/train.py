"""
train.py
--------
Fine-tuning pipeline for Uzbek sentiment analysis.
Two-phase: freeze backbone → unfreeze for end-to-end training.

Usage:
    python src/train.py --model xlmroberta --epochs 5 --batch_size 16
"""

import os
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model
from evaluate import compute_metrics, per_domain_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        type=str,   default="xlmroberta",
                   choices=["xlmroberta", "mbert"])
    p.add_argument("--data_dir",     type=str,   default="data/processed")
    p.add_argument("--save_dir",     type=str,   default="models/best_model")
    p.add_argument("--results_dir",  type=str,   default="results")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--freeze_epochs",type=int,   default=2)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--max_length",   type=int,   default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--patience",     type=int,   default=3)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss, preds, labels = 0, [], []

    for batch in tqdm(loader, desc=f"[Epoch {epoch}] Train", leave=False):
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        lbl    = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * ids.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbl.cpu().tolist())

    metrics = compute_metrics(labels, preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, split="Val"):
    model.eval()
    total_loss, preds, labels, domains = 0, [], [], []

    for batch in tqdm(loader, desc=f"[Epoch {epoch}] {split}", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["label"].to(device)

        logits = model(ids, mask)
        total_loss += criterion(logits, lbl).item() * ids.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbl.cpu().tolist())
        domains.extend(batch["domain"])

    metrics = compute_metrics(labels, preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, labels, preds, domains


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Uzbek Sentiment Analysis — Training")
    print(f"  Model: {args.model} | Device: {device}")
    print(f"{'='*55}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    tok_map = {
        "xlmroberta": "xlm-roberta-base",
        "mbert":      "bert-base-multilingual-cased",
    }

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        tokenizer_name=tok_map[args.model],
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    model = build_model(args.model, dropout=args.dropout).to(device)

    # Phase 1: freeze backbone
    model.freeze_backbone()
    class_weights = train_loader.dataset.get_class_weights().to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    total_steps   = len(train_loader) * args.epochs
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    log_path   = Path(args.results_dir) / "training_log.csv"
    log_fields = ["epoch","phase","train_loss","train_acc","train_f1",
                  "val_loss","val_acc","val_f1"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_f1, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):

        # Phase transition
        if epoch == args.freeze_epochs + 1:
            print(f"\n[Phase 2] Unfreezing backbone at epoch {epoch}")
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                int((args.epochs - args.freeze_epochs) * len(train_loader) * args.warmup_ratio),
                (args.epochs - args.freeze_epochs) * len(train_loader),
            )

        phase = "frozen" if epoch <= args.freeze_epochs else "full"
        t0    = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_m, val_labels, val_preds, val_domains = evaluate(
            model, val_loader, criterion, device, epoch
        )
        domain_m = per_domain_metrics(val_labels, val_preds, val_domains)

        print(
            f"Epoch {epoch:02d}/{args.epochs} [{phase}] | "
            f"Train Acc: {train_m['accuracy']:.4f} | "
            f"Val Acc: {val_m['accuracy']:.4f} F1: {val_m['f1']:.4f} | "
            f"{time.time()-t0:.1f}s"
        )
        for dom, dm in domain_m.items():
            print(f"  [{dom}] Acc: {dm['accuracy']:.4f} F1: {dm['f1']:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch": epoch, "phase": phase,
                "train_loss": round(train_m["loss"],4),
                "train_acc":  round(train_m["accuracy"],4),
                "train_f1":   round(train_m["f1"],4),
                "val_loss":   round(val_m["loss"],4),
                "val_acc":    round(val_m["accuracy"],4),
                "val_f1":     round(val_m["f1"],4),
            })

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            patience_counter = 0
            model.save_pretrained(args.save_dir)
            print(f"  ✓ Best model saved (Val F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stop] No improvement for {args.patience} epochs.")
                break

    print(f"\nTraining complete! Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
