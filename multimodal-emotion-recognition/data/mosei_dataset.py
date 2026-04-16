"""
data/mosei_dataset.py
Data loading for CMU-MOSEI with pre-extracted multimodal features.

MMSA-format pkl structure (verified):
  train/valid/test each contain:
    - text:                  (n, 50, 768)  — BERT embeddings per word
    - audio:                 (n, 50, 74)   — COVAREP features per time step
    - vision:                (n, 50, 35)   — OpenFace features per frame
    - raw_text:              (n,)          — original text strings
    - regression_labels:     (n,)          — sentiment intensity [-3, +3]
    - classification_labels: (n,)          — pre-computed 3-class: 0=neg, 1=neu, 2=pos
    - id:                    (n,)          — video segment IDs

Train: 16,326 | Valid: 1,871 | Test: 4,659
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class MOSEIDataset(Dataset):
    """
    PyTorch Dataset for MOSEI multimodal features.
    
    Can return all modalities or a specific one, controlled by
    the 'modality' argument. This lets us use the same dataset
    class for single-modality baselines AND the fused model.
    """
    
    def __init__(self, text, audio, vision, labels, modality="all"):
        self.text = torch.FloatTensor(text)
        self.audio = torch.FloatTensor(audio)
        self.vision = torch.FloatTensor(vision)
        self.labels = torch.LongTensor(labels)
        self.modality = modality
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {"label": self.labels[idx]}
        
        if self.modality == "text" or self.modality == "all":
            sample["text"] = self.text[idx]
        if self.modality == "audio" or self.modality == "all":
            sample["audio"] = self.audio[idx]
        if self.modality == "video" or self.modality == "all":
            sample["vision"] = self.vision[idx]
        
        return sample


def load_mosei_data(data_path):
    """
    Load MOSEI aligned pkl and extract features + labels.
    """
    print(f"Loading MOSEI data from {data_path}...")
    print("(This may take a moment for 4GB file)")
    
    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)
    
    splits = {}
    for split_name in ["train", "valid", "test"]:
        split = raw_data[split_name]
        
        text = np.array(split["text"], dtype=np.float32)
        audio = np.array(split["audio"], dtype=np.float32)
        vision = np.array(split["vision"], dtype=np.float32)
        labels = np.array(split["classification_labels"], dtype=np.int64)
        
        # Clean NaN values
        text = np.nan_to_num(text, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        vision = np.nan_to_num(vision, nan=0.0, posinf=0.0, neginf=0.0)
        
        splits[split_name] = {
            "text": text,
            "audio": audio,
            "vision": vision,
            "labels": labels,
        }
        
        print(f"  {split_name}: {len(labels)} samples | "
              f"text {text.shape} | audio {audio.shape} | vision {vision.shape}")
    
    del raw_data
    return splits


def get_class_weights(labels, num_classes, class_names):
    """Compute inverse-frequency class weights."""
    label_counts = Counter(labels)
    total = len(labels)
    
    print(f"\nClass distribution (train):")
    for i, name in enumerate(class_names):
        count = label_counts.get(i, 0)
        print(f"  {name:<12} {count:>6} ({count/total*100:.1f}%)")
    
    class_weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total / (num_classes * count)
        class_weights.append(weight)
    
    return torch.FloatTensor(class_weights)


def prepare_mosei_data(config):
    """
    Full data preparation pipeline.
    Returns splits dict and class_weights tensor.
    """
    splits = load_mosei_data(config["data_path"])
    class_weights = get_class_weights(
        splits["train"]["labels"],
        config["num_classes"],
        config["class_names"],
    )
    return splits, class_weights


def get_mosei_dataloaders(splits, config, modality="all"):
    """Create DataLoaders for a specific modality or all modalities."""
    loaders = {}
    for split_name in ["train", "valid", "test"]:
        if split_name not in splits:
            continue
        
        ds = MOSEIDataset(
            text=splits[split_name]["text"],
            audio=splits[split_name]["audio"],
            vision=splits[split_name]["vision"],
            labels=splits[split_name]["labels"],
            modality=modality,
        )
        
        loaders[split_name] = DataLoader(
            ds,
            batch_size=config["batch_size"],
            shuffle=(split_name == "train"),
            num_workers=0,
            drop_last=False,
        )
    
    return loaders
