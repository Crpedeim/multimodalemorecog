"""
models/text_encoder_mosei.py
Text branch for MOSEI — operates on PRE-EXTRACTED BERT embeddings.

This is different from the GoEmotions text encoder:
  - GoEmotions: Raw text → BERT (trainable) → projection → classifier
  - MOSEI: Pre-extracted BERT embeddings (768-dim) → LSTM → attention → classifier

Since BERT feature extraction is already done, this encoder learns how
to aggregate the per-word embeddings into a single utterance-level
representation and map it to emotion classes.

The LSTM captures how word-level emotions build up across the sentence.
"I was happy until they ruined everything" — the LSTM can learn that
the ending sentiment overrides the beginning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Attention pooling over sequence — same as audio/vision."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x, mask=None):
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled, weights


class TextEmotionEncoderMOSEI(nn.Module):
    """
    LSTM + Attention encoder for pre-extracted text features.
    
    Same architecture as audio and vision encoders — ensures
    fair comparison and consistent feature dimensionality for fusion.
    """
    
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2,
                 dropout=0.3, num_labels=3):
        super().__init__()
        
        # Project BERT 768-dim down to hidden_dim
        # This compression forces the model to keep only what's
        # relevant for emotion, discarding linguistic details
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.attention_pool = AttentionPool(hidden_dim * 2)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.hidden_dim = hidden_dim
    
    def extract_features(self, text):
        """Returns projected features for fusion."""
        x = self.input_proj(text)
        lstm_out, _ = self.lstm(x)
        pooled, attn_weights = self.attention_pool(lstm_out)
        features = self.projection(pooled)
        return features, attn_weights
    
    def forward(self, text):
        """Full forward pass with classification."""
        features, attn_weights = self.extract_features(text)
        logits = self.classifier(features)
        return logits, features, attn_weights
