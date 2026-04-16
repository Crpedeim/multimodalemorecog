"""
scripts/train_mosei_baselines.py
Train text-only, audio-only, and video-only baselines on MOSEI.

Usage:
    python scripts/train_mosei_baselines.py
    python scripts/train_mosei_baselines.py --data_path path/to/aligned_50.pkl
"""

import sys
import os
import argparse
import random
import time
import json

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.mosei_config import MOSEI_CONFIG
from data.mosei_dataset import prepare_mosei_data, get_mosei_dataloaders
from models.text_encoder_mosei import TextEmotionEncoderMOSEI
from models.audio_encoder import AudioEmotionEncoder
from models.vision_encoder import VisionEmotionEncoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, criterion, device, modality, max_grad_norm):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        if modality == "text":
            features = batch["text"].to(device)
        elif modality == "audio":
            features = batch["audio"].to(device)
        elif modality == "video":
            features = batch["vision"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _, _ = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, modality):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        if modality == "text":
            features = batch["text"].to(device)
        elif modality == "audio":
            features = batch["audio"].to(device)
        elif modality == "video":
            features = batch["vision"].to(device)

        labels = batch["label"].to(device)
        logits, _, _ = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": total_loss / n_batches,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "predictions": all_preds,
        "labels": all_labels,
    }


def train_and_evaluate(model, loaders, config, device, modality, save_dir, class_weights):
    """Full train → validate → test pipeline for one modality."""
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    print(f"\n{'='*55}")
    print(f"  Training {modality.upper()} encoder ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"{'='*55}")

    for epoch in range(config["epochs"]):
        start = time.time()

        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            device, modality, config["max_grad_norm"]
        )
        val_results = evaluate(model, loaders["valid"], criterion, device, modality)
        scheduler.step(val_results["f1_weighted"])

        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_results["loss"])
        history["val_f1"].append(val_results["f1_weighted"])
        history["val_acc"].append(val_results["accuracy"])

        is_best = val_results["f1_weighted"] > best_f1
        if is_best:
            best_f1 = val_results["f1_weighted"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": best_f1,
            }, os.path.join(save_dir, "best_model.pt"))
        else:
            patience_counter += 1

        marker = " ★" if is_best else ""
        print(f"  Epoch {epoch+1:>2}/{config['epochs']} ({elapsed:.1f}s) | "
              f"Train: {train_loss:.4f} | "
              f"Val F1: {val_results['f1_weighted']:.4f} | "
              f"Val Acc: {val_results['accuracy']:.4f}{marker}")

        if patience_counter >= config["patience"]:
            print(f"  Early stopping (no improvement for {config['patience']} epochs)")
            break

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation ──
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_results = evaluate(
        model, loaders["test"], nn.CrossEntropyLoss(), device, modality
    )

    print(f"\n  TEST ({modality.upper()}):")
    print(f"    Accuracy:      {test_results['accuracy']:.4f}")
    print(f"    F1 (weighted): {test_results['f1_weighted']:.4f}")
    print(f"    F1 (macro):    {test_results['f1_macro']:.4f}")
    print(f"\n{classification_report(test_results['labels'], test_results['predictions'], target_names=config['class_names'], digits=4)}")

    # Save test results
    cm = confusion_matrix(test_results["labels"], test_results["predictions"])
    report = {
        "modality": modality,
        "accuracy": float(test_results["accuracy"]),
        "f1_weighted": float(test_results["f1_weighted"]),
        "f1_macro": float(test_results["f1_macro"]),
        "confusion_matrix": cm.tolist(),
        "per_class": classification_report(
            test_results["labels"], test_results["predictions"],
            target_names=config["class_names"], digits=4, output_dict=True
        ),
    }
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(report, f, indent=2)

    return test_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    config = MOSEI_CONFIG.copy()
    if args.data_path:
        config["data_path"] = args.data_path

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ──
    splits, class_weights = prepare_mosei_data(config)

    # ── Train each modality ──
    results_table = {}

    for modality, ModelClass, input_dim, save_dir in [
        ("text",  TextEmotionEncoderMOSEI, config["text_dim"],  config["save_dir_text"]),
        ("audio", AudioEmotionEncoder,     config["audio_dim"], config["save_dir_audio"]),
        ("video", VisionEmotionEncoder,    config["video_dim"], config["save_dir_video"]),
    ]:
        model = ModelClass(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            num_labels=config["num_classes"],
        ).to(device)

        loaders = get_mosei_dataloaders(splits, config, modality=modality)

        test_results = train_and_evaluate(
            model, loaders, config, device, modality, save_dir, class_weights
        )

        results_table[modality] = {
            "f1_weighted": float(test_results["f1_weighted"]),
            "f1_macro": float(test_results["f1_macro"]),
            "accuracy": float(test_results["accuracy"]),
        }

    # ── Comparison ──
    print(f"\n{'='*60}")
    print(f"  SINGLE-MODALITY BASELINE COMPARISON (MOSEI)")
    print(f"{'='*60}")
    print(f"  {'Modality':<10} {'F1 (weighted)':>14} {'F1 (macro)':>12} {'Accuracy':>10}")
    print(f"  {'-'*46}")

    best_mod, best_f1 = "", 0
    for mod, m in results_table.items():
        if m["f1_weighted"] > best_f1:
            best_f1 = m["f1_weighted"]
            best_mod = mod
        print(f"  {mod:<10} {m['f1_weighted']:>14.4f} {m['f1_macro']:>12.4f} {m['accuracy']:>10.4f}")

    print(f"\n  Best single modality: {best_mod.upper()} ({best_f1:.4f})")
    print(f"  Fusion model must beat {best_f1:.4f} to justify multi-modal approach.")

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_comparison.json", "w") as f:
        json.dump(results_table, f, indent=2)

    print(f"\n  All results saved. Next: cross-modal attention fusion!")


if __name__ == "__main__":
    main()
