"""
models/audio_encoder.py
Audio branch of the multi-modal emotion recognition system.

Input: COVAREP features (74-dim per time step)
  - Fundamental frequency (F0) — pitch: higher when angry, lower when sad
  - Energy — loudness: high for anger/excitement, low for sadness
  - Spectral features — frequency distribution of the voice
  - Voice quality — jitter, shimmer (voice trembles when fearful/sad)
  - Voicing probability — is the person speaking or silent?

Architecture: LSTM → attention pooling → projection → classifier

Why LSTM over a simple MLP?
  Audio features are SEQUENTIAL — the way someone's voice changes over
  time matters. Starting calm and getting louder signals rising anger.
  An MLP that just averages features across time would lose this.

Why attention pooling?
  Not every time step is equally important. A 5-second clip might have
  4 seconds of normal speech and 1 second of emotional peak. Attention
  learns to weight the emotionally relevant moments more heavily.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    Attention-based pooling over a sequence.
    
    Instead of just averaging all time steps (which dilutes the signal),
    this learns to weight time steps by importance. The attention weights
    are interpretable — you can visualize which moments in the audio
    the model considers most emotionally relevant.
    
    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, hidden_dim)
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, hidden_dim)
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        
        return pooled, weights


class AudioEmotionEncoder(nn.Module):
    """
    LSTM + Attention encoder for COVAREP audio features.
    
    Args:
        input_dim:  Feature dimension per time step (74 for COVAREP)
        hidden_dim: LSTM hidden size and output projection size
        num_layers: Number of LSTM layers
        dropout:    Dropout probability
        num_labels: Number of emotion classes
    """
    
    def __init__(self, input_dim=74, hidden_dim=128, num_layers=2,
                 dropout=0.3, num_labels=3):
        super().__init__()
        
        # Input projection — normalize and project raw features
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Bidirectional LSTM — captures both forward and backward context
        # "What comes after this moment" matters as much as "what came before"
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention pooling over time steps
        # bidirectional LSTM outputs 2*hidden_dim
        self.attention_pool = AttentionPool(hidden_dim * 2)
        
        # Projection to shared feature space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.hidden_dim = hidden_dim
    
    def extract_features(self, audio):
        """
        Returns projected features (for fusion module).
        
        Args:
            audio: (batch, seq_len, 74)
        Returns:
            features: (batch, hidden_dim)
        """
        # Project input
        x = self.input_proj(audio)  # (batch, seq_len, hidden_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)
        
        # Attention pooling
        pooled, attn_weights = self.attention_pool(lstm_out)  # (batch, hidden_dim * 2)
        
        # Project to shared space
        features = self.projection(pooled)  # (batch, hidden_dim)
        
        return features, attn_weights
    
    def forward(self, audio):
        """
        Full forward pass with classification.
        
        Returns:
            logits: (batch, num_labels)
            features: (batch, hidden_dim)
            attn_weights: (batch, seq_len) — for visualization
        """
        features, attn_weights = self.extract_features(audio)
        logits = self.classifier(features)
        return logits, features, attn_weights
