"""
predict.py
----------
Run inference on any Uzbek text — single input or batch CSV.

Usage:
    python src/predict.py --text "Bu restoran juda yaxshi edi!"
    python src/predict.py --csv data/samples/sample_texts.csv
"""

import argparse
import os
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer

from model import TransformerModel
from dataset import LABEL_EMOJI, ID2LABEL

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
MAX_LENGTH  = 128


def load_model(model_dir="models/best_model", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = TransformerModel(model_name=model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.classifier.load_state_dict(
        torch.load(os.path.join(model_dir, "classifier_head.pt"), map_location=device)
    )
    model.eval()
    print(f"[Predict] Model ready on {device}")
    return model, tokenizer, device


def predict_text(text, model, tokenizer, device) -> dict:
    enc  = tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                     truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(probs.argmax())
    return {
        "prediction":    LABEL_NAMES[idx],
        "confidence":    float(probs[idx]),
        "prob_negative": float(probs[0]),
        "prob_neutral":  float(probs[1]),
        "prob_positive": float(probs[2]),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text",      type=str, default=None)
    p.add_argument("--csv",       type=str, default=None)
    p.add_argument("--output",    type=str, default="results/predictions.csv")
    p.add_argument("--model_dir", type=str, default="models/best_model")
    return p.parse_args()


def main():
    args = parse_args()
    model, tokenizer, device = load_model(args.model_dir)

    if args.text:
        r = predict_text(args.text, model, tokenizer, device)
        emoji = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}[r["prediction"]]
        print(f"\n{'='*48}")
        print(f"  Text:       {args.text[:80]}")
        print(f"  Prediction: {emoji} {r['prediction']}")
        print(f"  Confidence: {r['confidence']:.1%}")
        print(f"  ─────────────────────────────────")
        print(f"  😠 Negative: {r['prob_negative']:.1%}")
        print(f"  😐 Neutral:  {r['prob_neutral']:.1%}")
        print(f"  😊 Positive: {r['prob_positive']:.1%}")
        print(f"{'='*48}\n")

    elif args.csv:
        df      = pd.read_csv(args.csv)
        results = []
        for _, row in df.iterrows():
            r = predict_text(str(row["text"]), model, tokenizer, device)
            results.append(r)
            emoji = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}[r["prediction"]]
            print(f"  {emoji} {r['prediction']:<10} ({r['confidence']:.1%})  |  {str(row['text'])[:55]}")

        out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        os.makedirs(Path(args.output).parent, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\n[Results] Saved → {args.output}")
    else:
        print("[Error] Provide --text or --csv")


if __name__ == "__main__":
    main()
