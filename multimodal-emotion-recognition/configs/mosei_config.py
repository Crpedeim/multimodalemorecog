"""
configs/mosei_config.py
Configuration for CMU-MOSEI multimodal experiments.

MOSEI contains pre-extracted features:
  - Text: 768-dim BERT embeddings
  - Audio: 74-dim COVAREP features (pitch, energy, spectral, voice quality)
  - Video: 35-dim OpenFace features (facial action units, head pose, gaze)

Labels: Sentiment intensity (-3 to +3), which we convert to 3-class:
  Negative (< 0) | Neutral (== 0) | Positive (> 0)

This maps to emotion valence — a standard dimension in affective computing.
Negative valence = anger, sadness, fear, disgust
Positive valence = joy, excitement, surprise
"""

MOSEI_CONFIG = {
    # Data
    "data_path": "data/mosei_aligned_50.pkl",  # Update this to your actual path
    "num_classes": 3,       # negative, neutral, positive
    "class_names": ["negative", "neutral", "positive"],
    "max_seq_len": 50,      # Max sequence length (aligned to 50 in standard preprocessing)

    # Feature dimensions (fixed by the pre-extraction pipeline)
    "text_dim": 768,        # BERT hidden size
    "audio_dim": 74,        # COVAREP features
    "video_dim": 35,        # OpenFace features

    # Model architecture
    "hidden_dim": 128,      # Shared hidden dim for all encoders
    "dropout": 0.3,
    "num_layers": 2,        # For LSTM/GRU encoders

    # Training
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 20,           # Small models converge fast
    "patience": 5,          # Early stopping patience
    "max_grad_norm": 1.0,

    # Paths
    "save_dir_text": "results/mosei_text_baseline",
    "save_dir_audio": "results/mosei_audio_baseline",
    "save_dir_video": "results/mosei_video_baseline",

    # Device
    "seed": 42,
}
