"""
models/fusion.py
Cross-Modal Attention Fusion Module — the core contribution of this project.

This module implements two fusion strategies:
  1. Naive Concatenation (baseline) — stack features, classify
  2. Cross-Modal Attention (main) — each modality attends to others

The key insight: modality importance varies per sample.
  - "I HATE this" → text is unambiguous, audio/video add noise
  - "yeah it was fine" → text is ambiguous, need audio tone + facial expression

Cross-modal attention learns these dynamic weightings automatically
through the Query-Key-Value mechanism from transformers.

Architecture:
  Text features ─┐
                  ├─→ Cross-Modal Attention ─→ Gated Aggregation ─→ Classifier
  Audio features ─┤      (T→A, T→V, A→T,
                  │       A→V, V→T, V→A)
  Video features ─┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionLayer(nn.Module):
    """
    Single cross-modal attention: one modality attends to another.
    
    Given a 'query' modality and a 'context' modality:
    - Query modality asks: "what information from the other modality is relevant to me?"
    - Context modality provides Keys and Values
    - Output is the query modality enriched with relevant context information
    
    This is exactly the multi-head attention from "Attention Is All You Need"
    but applied ACROSS modalities instead of within a single sequence.
    
    Args:
        d_model: Feature dimension (must be same for both modalities)
        num_heads: Number of attention heads (each head attends to different aspects)
        dropout: Attention dropout
    """
    
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, context):
        """
        Args:
            query:   (batch, d_model) — the modality asking for information
            context: (batch, d_model) — the modality providing information
        
        Returns:
            enriched: (batch, d_model) — query enriched with context info
            attn_weights: (batch, 1, 1) — how much query attended to context
        """
        # Reshape to (batch, seq_len=1, d_model) for MultiheadAttention
        q = query.unsqueeze(1)
        k = context.unsqueeze(1)
        v = context.unsqueeze(1)
        
        attended, attn_weights = self.attention(q, k, v)
        attended = attended.squeeze(1)  # (batch, d_model)
        
        # Residual connection + layer norm
        # The query modality keeps its own information + adds what it learned
        enriched = self.norm(query + self.dropout(attended))
        
        return enriched, attn_weights


class GatedFusion(nn.Module):
    """
    Gated aggregation of enriched modality features.
    
    Instead of just concatenating, learns a soft gate that weights
    each modality's contribution. The gate values are interpretable:
    you can see which modality the model relied on most for each sample.
    
    gate = sigmoid(W · [text; audio; video])
    output = gate_t * text + gate_a * audio + gate_v * video
    
    Args:
        d_model: Feature dimension per modality
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),  # 3 gate values, one per modality
            nn.Softmax(dim=-1),     # Gates sum to 1
        )
    
    def forward(self, text_feat, audio_feat, video_feat):
        """
        Args:
            text_feat, audio_feat, video_feat: each (batch, d_model)
        
        Returns:
            fused: (batch, d_model) — weighted combination
            gate_weights: (batch, 3) — [text_weight, audio_weight, video_weight]
        """
        concat = torch.cat([text_feat, audio_feat, video_feat], dim=-1)
        gate_weights = self.gate_net(concat)  # (batch, 3)
        
        fused = (
            gate_weights[:, 0:1] * text_feat +
            gate_weights[:, 1:2] * audio_feat +
            gate_weights[:, 2:3] * video_feat
        )
        
        return fused, gate_weights


class CrossModalAttentionFusion(nn.Module):
    """
    Full cross-modal attention fusion model.
    
    Each modality attends to both other modalities, then the enriched
    representations are combined through gated fusion.
    
    Cross-attention pairs:
      Text  → Audio  (does the voice confirm the words?)
      Text  → Video  (does the face match the sentiment?)
      Audio → Text   (do the words explain this tone?)
      Audio → Video  (does the face match this voice?)
      Video → Text   (do the words explain this expression?)
      Video → Audio  (does the voice match this expression?)
    
    Args:
        d_model: Feature dimension (shared across all modalities)
        num_heads: Number of attention heads
        num_classes: Output classes
        dropout: Dropout probability
    """
    
    def __init__(self, d_model=128, num_heads=4, num_classes=3, dropout=0.3):
        super().__init__()
        
        # Cross-modal attention layers (6 pairs)
        self.text_attend_audio = CrossModalAttentionLayer(d_model, num_heads, dropout)
        self.text_attend_video = CrossModalAttentionLayer(d_model, num_heads, dropout)
        
        self.audio_attend_text = CrossModalAttentionLayer(d_model, num_heads, dropout)
        self.audio_attend_video = CrossModalAttentionLayer(d_model, num_heads, dropout)
        
        self.video_attend_text = CrossModalAttentionLayer(d_model, num_heads, dropout)
        self.video_attend_audio = CrossModalAttentionLayer(d_model, num_heads, dropout)
        
        # Combine the two cross-attended versions per modality
        self.text_combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.audio_combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.video_combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Gated fusion
        self.gate = GatedFusion(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
    
    def forward(self, text_feat, audio_feat, video_feat):
        """
        Args:
            text_feat:  (batch, d_model) — from text encoder
            audio_feat: (batch, d_model) — from audio encoder
            video_feat: (batch, d_model) — from video encoder
        
        Returns:
            logits: (batch, num_classes)
            gate_weights: (batch, 3) — modality importance per sample
            cross_attn_info: dict of attention weights for visualization
        """
        # ── Cross-modal attention ──
        # Text enriched by audio and video
        t_from_a, attn_ta = self.text_attend_audio(text_feat, audio_feat)
        t_from_v, attn_tv = self.text_attend_video(text_feat, video_feat)
        
        # Audio enriched by text and video
        a_from_t, attn_at = self.audio_attend_text(audio_feat, text_feat)
        a_from_v, attn_av = self.audio_attend_video(audio_feat, video_feat)
        
        # Video enriched by text and audio
        v_from_t, attn_vt = self.video_attend_text(video_feat, text_feat)
        v_from_a, attn_va = self.video_attend_audio(video_feat, audio_feat)
        
        # ── Combine cross-attended features per modality ──
        # Each modality now has information from both other modalities
        text_enriched = self.text_combine(torch.cat([t_from_a, t_from_v], dim=-1))
        audio_enriched = self.audio_combine(torch.cat([a_from_t, a_from_v], dim=-1))
        video_enriched = self.video_combine(torch.cat([v_from_t, v_from_a], dim=-1))
        
        # ── Gated fusion ──
        fused, gate_weights = self.gate(text_enriched, audio_enriched, video_enriched)
        
        # ── Classification ──
        logits = self.classifier(fused)
        
        # Collect attention info for visualization
        cross_attn_info = {
            "text_attend_audio": attn_ta,
            "text_attend_video": attn_tv,
            "audio_attend_text": attn_at,
            "audio_attend_video": attn_av,
            "video_attend_text": attn_vt,
            "video_attend_audio": attn_va,
        }
        
        return logits, gate_weights, cross_attn_info


class NaiveConcatFusion(nn.Module):
    """
    Baseline fusion: just concatenate all three feature vectors.
    
    This exists as a comparison point. If cross-modal attention
    doesn't beat this, the attention mechanism isn't helping.
    If it does beat this (which it should), it proves that
    dynamic modality weighting matters.
    """
    
    def __init__(self, d_model=128, num_classes=3, dropout=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
    
    def forward(self, text_feat, audio_feat, video_feat):
        concat = torch.cat([text_feat, audio_feat, video_feat], dim=-1)
        logits = self.classifier(concat)
        return logits, None, None
