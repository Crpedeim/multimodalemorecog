"""
models/vision_encoder.py
Vision branch using a Temporal Transformer over OpenFace features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class VisionEmotionEncoder(nn.Module):
    """
    Temporal Transformer encoder for OpenFace visual features.
    """
    def __init__(self, input_dim=35, hidden_dim=128, num_layers=2,
                 dropout=0.3, num_labels=3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling to collapse sequence to a single vector
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.hidden_dim = hidden_dim
    
    def extract_features(self, vision):
        x = self.input_proj(vision)
        x = self.pos_encoder(x)
        transformer_out = self.transformer(x)
        
        # Pool the sequence using attention
        scores = self.attention(transformer_out).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        features = torch.bmm(weights.unsqueeze(1), transformer_out).squeeze(1)
        
        return features, weights
    
    def forward(self, vision):
        features, attn_weights = self.extract_features(vision)
        logits = self.classifier(features)
        return logits, features, attn_weights