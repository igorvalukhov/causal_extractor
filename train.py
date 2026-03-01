"""
Обучение BiLSTM-CRF для извлечения причинно-следственных связей.
v1: train/val/test имеют непересекающиеся словари сущностей.

Запуск:
    python train.py --epochs 20 --batch_size 32 --samples 10000
"""

import argparse
import json
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

from data_generator import generate_dataset, save_dataset, load_dataset
from model import (
    BiLSTMCRF, CausalDataset, Vocabulary, collate_fn,
    TAG2ID, ID2TAG, BIOES_TAGS,
)


# ──────────────────────────────────────────────────────────────────
# Метрики по именованным спанам
# ──────────────────────────────────────────────────────────────────

def extract_spans(tag_ids: List[int]) -> Dict[str, List[Tuple[int, int]]]:
    spans = defaultdict(list)
    i = 0
    while i < len(tag_ids):
        tag = ID2TAG.get(tag_ids[i], "O")
        if tag.startswith("S-"):
            spans[tag[2:]].append((i, i))
            i += 1
        elif tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < len(tag_ids):
                t = ID2TAG.get(tag_ids[j], "O")
                if t == f"E-{label}":
                    spans[label].append((i, j))
                    break
                elif not t.startswith("I-"):
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return spans


def compute_metrics(pred_all, true_all):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for pred, true in zip(pred_all, true_all):
        for label in ["CAUSE", "EFFECT"]:
            p = set(extract_spans(pred).get(label, []))
            t = set(extract_spans(true).get(label, []))
            tp[label] += len(p & t)
            fp[label] += len(p - t)
            fn[label] += len(t - p)
    metrics = {}
    for label in ["CAUSE", "EFFECT"]:
        pr = tp[label] / (tp[label] + fp[label] + 1e-9)
        re = tp[label] / (tp[label] + fn[label] + 1e-9)
        f1 = 2 * pr * re / (pr + re + 1e-9)
        metrics[f"{label}_P"]  = round(pr, 4)
        metrics[f"{label}_R"]  = round(re, 4)
        metrics[f"{label}_F1"] = round(f1, 4)
    metrics["macro_F1"] = round(
        (metrics["CAUSE_F1"] + metrics["EFFECT_F1"]) / 2, 4)
    return metrics


# ──────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    pred_all, true_all = [], []
    with torch.no_grad():
        for tokens, tags, lengths in loader:
            tokens, tags, lengths = tokens.to(device), tags.to(device), lengths.to(device)
            loss = model(tokens, tags, lengths)
            total_loss += loss.item()
            preds = model.predict(tokens, lengths)
            for pred, tag_row, length in zip(preds, tags, lengths):
                true_all.append(tag_row[:length].tolist())
                pred_all.append(pred)
    return total_loss / len(loader), compute_metrics(pred_all, true_all)


# ──────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tr_path  = data_dir / "train.json"
    val_path = data_dir / "val.json"
    te_path  = data_dir / "test.json"

    if not tr_path.exists() or args.regenerate:
        print(f"Генерируем {args.samples} примеров (split vocab)...")
        random.seed(42)
        train_data, val_data, test_data = generate_dataset(args.samples, neg_ratio=0.2)
        save_dataset(train_data, str(tr_path))
        save_dataset(val_data,   str(val_path))
        save_dataset(test_data,  str(te_path))
    else:
        train_data = load_dataset(str(tr_path))
        val_data   = load_dataset(str(val_path))
        test_data  = load_dataset(str(te_path))
        print(f"Загружено: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    # Словарь строим только по train
    vocab = Vocabulary()
    vocab.build(train_data, min_freq=1)
    with open(data_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    train_ds = CausalDataset(train_data, vocab)
    val_ds   = CausalDataset(val_data,   vocab)
    test_ds  = CausalDataset(test_data,  vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = BiLSTMCRF(
        vocab_size=len(vocab),
        num_tags=len(BIOES_TAGS),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5)

    best_f1    = 0.0
    model_path = data_dir / "best_model.pt"
    history    = []

    print("\n═══ Начало обучения ═══\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (tokens, tags, lengths) in enumerate(train_loader):
            tokens, tags, lengths = tokens.to(device), tags.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss = model(tokens, tags, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)}"
                      f" | Loss: {loss.item():.4f}", end="\r")

        train_loss /= len(train_loader)
        val_loss, val_m = evaluate(model, val_loader, device)
        macro_f1 = val_m["macro_F1"]
        scheduler.step(macro_f1)

        print(f"\nEpoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Macro F1: {macro_f1:.4f} | "
              f"CAUSE F1: {val_m['CAUSE_F1']:.4f} | "
              f"EFFECT F1: {val_m['EFFECT_F1']:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, **val_m})

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"model_state": model.state_dict(),
                         "vocab": vocab, "args": vars(args)}, model_path)
            print(f"  ✓ Лучшая модель сохранена (F1={best_f1:.4f})")

    # Финальный тест на отложенных данных с новыми фразами
    print("\n═══ Тест (unseen entities) ═══")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _, test_m = evaluate(model, test_loader, device)
    for k, v in test_m.items():
        print(f"  {k}: {v}")

    with open(data_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\nМодель сохранена: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--samples",    type=int,   default=5000)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--embed_dim",  type=int,   default=64)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    train(args)
