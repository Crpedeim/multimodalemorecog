"""
scripts/train_fusion.py
Train the fusion models on MOSEI using pre-trained encoder features.

Trains two models:
  1. Naive Concatenation (baseline fusion)
  2. Cross-Modal Attention (main contribution)

Compares both against single-modality baselines to show improvement.

Usage:
    python scripts/train_fusion.py --data_path data/aligned_50.pkl
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
from models.multimodal_model import MultiModalEmotionModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0
    n_batches = 0
    all_gate_weights = []

    for batch in dataloader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, gate_weights, _ = model(text, audio, vision)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if gate_weights is not None:
            all_gate_weights.append(gate_weights.detach().cpu())

    avg_gates = None
    if all_gate_weights:
        avg_gates = torch.cat(all_gate_weights, dim=0).mean(dim=0).numpy()

    return total_loss / n_batches, avg_gates


# @torch.no_grad()
# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0
#     n_batches = 0
#     all_preds, all_labels = [], []
#     all_gate_weights = []

#     for batch in dataloader:
#         text = batch["text"].to(device)
#         audio = batch["audio"].to(device)
#         vision = batch["vision"].to(device)
#         labels = batch["label"].to(device)

#         logits, gate_weights, _ = model(text, audio, vision)
#         loss = criterion(logits, labels)

#         total_loss += loss.item()
#         n_batches += 1

#         preds = torch.argmax(logits, dim=-1)
#         all_preds = np.array(all_preds)
#         all_labels = np.array(all_labels)

#     avg_gates = None
#     per_class_gates = {} # New dictionary
    
#     if all_gate_weights:
#         gates_tensor = torch.cat(all_gate_weights, dim=0).numpy()
#         avg_gates = gates_tensor.mean(axis=0)
        
#         # Calculate gates for Negative (0), Neutral (1), Positive (2)
#         for c in range(3): 
#             mask = (all_labels == c)
#             if np.sum(mask) > 0:
#                 per_class_gates[int(c)] = gates_tensor[mask].mean(axis=0).tolist()

#     return {
#         "loss": total_loss / n_batches,
#         "accuracy": accuracy_score(all_labels, all_preds),
#         "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
#         "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
#         "predictions": all_preds,
#         "labels": all_labels,
#         "gate_weights": avg_gates,
#         "per_class_gates": per_class_gates, # Added to return dict
#     }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds, all_labels = [], []
    all_gate_weights = []

    for batch in dataloader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].to(device)

        logits, gate_weights, _ = model(text, audio, vision)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if gate_weights is not None:
            all_gate_weights.append(gate_weights.cpu())

    # --- THE SAFEGUARD BLOCK (MUST BE HERE, BEFORE METRICS) ---
    if n_batches == 0 or len(all_labels) == 0:
        print("  [Warning] Dataloader yielded 0 samples! Returning dummy scores to prevent crash.")
        return {
            "loss": 999.0,
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "f1_macro": 0.0,
            "predictions": np.array([]),
            "labels": np.array([]),
            "gate_weights": None,
            "per_class_gates": {}
        }
    # ----------------------------------------------------------

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_gates = None
    per_class_gates = {}
    
    if all_gate_weights:
        gates_tensor = torch.cat(all_gate_weights, dim=0).numpy()
        avg_gates = gates_tensor.mean(axis=0)
        
        for c in range(3): 
            mask = (all_labels == c)
            if np.sum(mask) > 0:
                per_class_gates[int(c)] = gates_tensor[mask].mean(axis=0).tolist()

    return {
        "loss": total_loss / n_batches,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "predictions": all_preds,
        "labels": all_labels,
        "gate_weights": avg_gates,
        "per_class_gates": per_class_gates, 
    }

def train_fusion_model(model, loaders, config, device, save_dir, class_weights, fusion_type):
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Separate parameters for different learning rates
    fusion_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'fusion' in name:
            fusion_params.append(param)
        else:
            encoder_params.append(param)
            
    optimizer = torch.optim.Adam([
        {"params": fusion_params, "lr": config["learning_rate"]},
        {"params": encoder_params, "lr": 1e-5}  # Tiny LR for pre-trained encoders
    ], weight_decay=config["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_f1 = 0.0
    patience_counter = 0
    max_patience = 7  # More patience for fusion since it's harder to learn
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": [], "gate_weights": []}

    fusion_name = "Cross-Modal Attention" if fusion_type == "attention" else "Naive Concatenation"
    print(f"\n{'='*55}")
    print(f"  Training {fusion_name} Fusion")
    print(f"{'='*55}")

    for epoch in range(30):  # More epochs for fusion
        start = time.time()

        train_loss, train_gates = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, config["max_grad_norm"]
        )
        val_results = evaluate(model, loaders["valid"], criterion, device)
        scheduler.step(val_results["f1_weighted"])

        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_results["loss"])
        history["val_f1"].append(val_results["f1_weighted"])
        history["val_acc"].append(val_results["accuracy"])
        if val_results["gate_weights"] is not None:
            history["gate_weights"].append(val_results["gate_weights"].tolist())

        is_best = val_results["f1_weighted"] > best_f1
        if is_best:
            best_f1 = val_results["f1_weighted"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "fusion_state_dict": model.fusion.state_dict(),
                "epoch": epoch,
                "val_f1": best_f1,
            }, os.path.join(save_dir, "best_model.pt"))
        else:
            patience_counter += 1

        marker = " ★" if is_best else ""
        gate_str = ""
        if val_results["gate_weights"] is not None:
            g = val_results["gate_weights"]
            gate_str = f" | Gates: T={g[0]:.2f} A={g[1]:.2f} V={g[2]:.2f}"

        print(f"  Epoch {epoch+1:>2} ({elapsed:.1f}s) | "
              f"Train: {train_loss:.4f} | "
              f"Val F1: {val_results['f1_weighted']:.4f}{gate_str}{marker}")

        if patience_counter >= max_patience:
            print(f"  Early stopping (no improvement for {max_patience} epochs)")
            break

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation ──
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_results = evaluate(model, loaders["test"], nn.CrossEntropyLoss(), device)

    print(f"\n  TEST ({fusion_name}):")
    print(f"    Accuracy:      {test_results['accuracy']:.4f}")
    print(f"    F1 (weighted): {test_results['f1_weighted']:.4f}")
    print(f"    F1 (macro):    {test_results['f1_macro']:.4f}")

    if test_results["gate_weights"] is not None:
        g = test_results["gate_weights"]
        print(f"    Gate weights:  Text={g[0]:.3f}  Audio={g[1]:.3f}  Video={g[2]:.3f}")

    class_names = config["class_names"]
    print(f"\n{classification_report(test_results['labels'], test_results['predictions'], target_names=class_names, digits=4)}")

    # Save results
    cm = confusion_matrix(test_results["labels"], test_results["predictions"])
    report = {
        "fusion_type": fusion_type,
        "accuracy": float(test_results["accuracy"]),
        "f1_weighted": float(test_results["f1_weighted"]),
        "f1_macro": float(test_results["f1_macro"]),
        "confusion_matrix": cm.tolist(),
        "gate_weights": test_results["gate_weights"].tolist() if test_results["gate_weights"] is not None else None,
        "per_class": classification_report(
            test_results["labels"], test_results["predictions"],
            target_names=class_names, digits=4, output_dict=True
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
    loaders = get_mosei_dataloaders(splits, config, modality="all")

    # ── Check for pre-trained encoder checkpoints ──
    text_ckpt = os.path.join(config["save_dir_text"], "best_model.pt")
    audio_ckpt = os.path.join(config["save_dir_audio"], "best_model.pt")
    video_ckpt = os.path.join(config["save_dir_video"], "best_model.pt")

    for path, name in [(text_ckpt, "Text"), (audio_ckpt, "Audio"), (video_ckpt, "Video")]:
        if not os.path.exists(path):
            print(f"\n  ERROR: {name} encoder checkpoint not found at {path}")
            print(f"  Run 'python scripts/train_mosei_baselines.py' first!")
            return
    
    print(f"\nAll encoder checkpoints found. Loading pre-trained encoders...")

    # ── Load single-modality baselines for comparison ──
    baseline_path = "results/baseline_comparison.json"
    baselines = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baselines = json.load(f)

    # ── Train Naive Concatenation ──
    print("\n" + "=" * 60)
    print("  STEP 1: Naive Concatenation Baseline")
    print("=" * 60)

    concat_model = MultiModalEmotionModel(
        config, fusion_type="concat",
        text_checkpoint=text_ckpt, audio_checkpoint=audio_ckpt, video_checkpoint=video_ckpt,
    ).to(device)

    concat_results = train_fusion_model(
        concat_model, loaders, config, device,
        "results/mosei_concat_fusion", class_weights, "concat"
    )

    # ── Train Cross-Modal Attention ──
    print("\n" + "=" * 60)
    print("  STEP 2: Cross-Modal Attention Fusion")
    print("=" * 60)

    set_seed(config["seed"])  # Reset seed for fair comparison

    attention_model = MultiModalEmotionModel(
        config, fusion_type="attention",
        text_checkpoint=text_ckpt, audio_checkpoint=audio_ckpt, video_checkpoint=video_ckpt,
        freeze_encoders=False # Set to False!
    ).to(device)

    attention_results = train_fusion_model(
        attention_model, loaders, config, device,
        "results/mosei_attention_fusion", class_weights, "attention"
    )

    # ── Final Comparison Table ──
    print(f"\n{'='*65}")
    print(f"  COMPLETE MODEL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Model':<30} {'F1 (weighted)':>14} {'F1 (macro)':>12} {'Accuracy':>10}")
    print(f"  {'-'*66}")

    # Single modality baselines
    if baselines:
        for mod in ["text", "audio", "video"]:
            if mod in baselines:
                m = baselines[mod]
                print(f"  {mod.capitalize() + ' only':<30} {m['f1_weighted']:>14.4f} {m['f1_macro']:>12.4f} {m['accuracy']:>10.4f}")
        print(f"  {'-'*66}")

    # Fusion models
    print(f"  {'Naive Concatenation':<30} {concat_results['f1_weighted']:>14.4f} {concat_results['f1_macro']:>12.4f} {concat_results['accuracy']:>10.4f}")
    print(f"  {'Cross-Modal Attention':<30} {attention_results['f1_weighted']:>14.4f} {attention_results['f1_macro']:>12.4f} {attention_results['accuracy']:>10.4f}")

    # Improvement
    text_f1 = baselines.get("text", {}).get("f1_weighted", 0)
    attn_f1 = attention_results["f1_weighted"]
    if text_f1 > 0:
        improvement = (attn_f1 - text_f1) / text_f1 * 100
        print(f"\n  Improvement over text-only: {improvement:+.2f}%")
        if attn_f1 > text_f1:
            print(f"  ✓ Cross-modal attention BEATS text-only baseline!")
        else:
            print(f"  ✗ Fusion did not beat text-only — investigate why.")

    # Gate weights analysis
    if attention_results.get("gate_weights") is not None:
        g = attention_results["gate_weights"]
        print(f"\n  Average global gate weights:")
        print(f"    Text: {g[0]:.3f} | Audio: {g[1]:.3f} | Video: {g[2]:.3f}")
        
        # New Per-Class Print Block
        print(f"\n  Per-Class Gate Analysis (Text, Audio, Video):")
        class_names = config.get("class_names", ["Negative", "Neutral", "Positive"])
        for c, gates in attention_results.get("per_class_gates", {}).items():
            if c < len(class_names):
                print(f"    {class_names[c]:<10}: {gates[0]:.3f}, {gates[1]:.3f}, {gates[2]:.3f}")

    # Save complete comparison
    complete_results = {
        "baselines": baselines,
        "concat_fusion": {
            "f1_weighted": float(concat_results["f1_weighted"]),
            "f1_macro": float(concat_results["f1_macro"]),
            "accuracy": float(concat_results["accuracy"]),
        },
        "attention_fusion": {
            "f1_weighted": float(attention_results["f1_weighted"]),
            "f1_macro": float(attention_results["f1_macro"]),
            "accuracy": float(attention_results["accuracy"]),
            "gate_weights": attention_results["gate_weights"].tolist() if attention_results["gate_weights"] is not None else None,
        },
    }
    with open("results/complete_comparison.json", "w") as f:
        json.dump(complete_results, f, indent=2)

    print(f"\n  All results saved to results/")
    print(f"  Next: ablation studies and attention visualizations!")


if __name__ == "__main__":
    main()
