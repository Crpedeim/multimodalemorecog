"""
models/multimodal_model.py
Complete multi-modal emotion recognition model.

Combines three pre-trained encoders with the fusion module.
Encoders are FROZEN during fusion training — we only train
the fusion layers. This:
  1. Prevents overfitting (don't re-train 2M+ encoder params)
  2. Ensures encoder features remain stable
  3. Makes training fast (only ~200K fusion params to learn)

Architecture:
  Raw features → [Frozen Encoders] → 128-dim per modality → Fusion → Classification
"""

import torch
import torch.nn as nn
from models.text_encoder_mosei import TextEmotionEncoderMOSEI
from models.audio_encoder import AudioEmotionEncoder
from models.vision_encoder import VisionEmotionEncoder
from models.fusion import CrossModalAttentionFusion, NaiveConcatFusion


class MultiModalEmotionModel(nn.Module):
    """
    End-to-end multi-modal emotion recognition model.
    
    Loads pre-trained encoder weights, freezes them,
    and trains only the fusion module on top.
    
    Args:
        config: Configuration dict with model hyperparameters
        fusion_type: "attention" or "concat"
        text_checkpoint: Path to pre-trained text encoder
        audio_checkpoint: Path to pre-trained audio encoder
        video_checkpoint: Path to pre-trained video encoder
    """
    
    def __init__(self, config, fusion_type="attention",
                 text_checkpoint=None, audio_checkpoint=None, video_checkpoint=None, freeze_encoders=False):
        super().__init__()
        
        d_model = config["hidden_dim"]
        
        # ── Encoders (will be frozen) ──
        self.text_encoder = TextEmotionEncoderMOSEI(
            input_dim=config["text_dim"], hidden_dim=d_model,
            num_layers=config["num_layers"], dropout=0.0,  # No dropout in frozen encoders
            num_labels=config["num_classes"],
        )
        self.audio_encoder = AudioEmotionEncoder(
            input_dim=config["audio_dim"], hidden_dim=d_model,
            num_layers=config["num_layers"], dropout=0.0,
            num_labels=config["num_classes"],
        )
        self.video_encoder = VisionEmotionEncoder(
            input_dim=config["video_dim"], hidden_dim=d_model,
            num_layers=config["num_layers"], dropout=0.0,
            num_labels=config["num_classes"],
        )
        
        # Load pre-trained weights
        if text_checkpoint:
            ckpt = torch.load(text_checkpoint, map_location="cpu")
            self.text_encoder.load_state_dict(ckpt["model_state_dict"])
        if audio_checkpoint:
            ckpt = torch.load(audio_checkpoint, map_location="cpu")
            self.audio_encoder.load_state_dict(ckpt["model_state_dict"])
        if video_checkpoint:
            ckpt = torch.load(video_checkpoint, map_location="cpu")
            self.video_encoder.load_state_dict(ckpt["model_state_dict"])
              
            
        
        # Conditionally freeze encoders
        if freeze_encoders:
            self._freeze_encoders()
        else:
            print("  Encoders UNFREEZED — training end-to-end")
        # ── Fusion module ──
        if fusion_type == "attention":
            self.fusion = CrossModalAttentionFusion(
                d_model=d_model,
                num_heads=4,
                num_classes=config["num_classes"],
                dropout=config["dropout"],
            )
        elif fusion_type == "concat":
            self.fusion = NaiveConcatFusion(
                d_model=d_model,
                num_classes=config["num_classes"],
                dropout=config["dropout"],
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.fusion_type = fusion_type
        
        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total:,} | Trainable (fusion only): {trainable:,}")
    
    def _freeze_encoders(self):
        """Freeze all encoder parameters."""
        for encoder in [self.text_encoder, self.audio_encoder, self.video_encoder]:
            for param in encoder.parameters():
                param.requires_grad = False
        print("  Encoders frozen — training fusion module only")
    
    def forward(self, text, audio, vision):
        """
        Args:
            text:   (batch, seq_len, 768)
            audio:  (batch, seq_len, 74)
            vision: (batch, seq_len, 35)
        
        Returns:
            logits: (batch, num_classes)
            gate_weights: (batch, 3) or None
            cross_attn_info: dict or None
        """
        # Extract features from frozen encoders
        text_feat, _ = self.text_encoder.extract_features(text)
        audio_feat, _ = self.audio_encoder.extract_features(audio)
        video_feat, _ = self.video_encoder.extract_features(vision)
        
        # Fuse
        logits, gate_weights, cross_attn_info = self.fusion(
            text_feat, audio_feat, video_feat
        )
        
        return logits, gate_weights, cross_attn_info
