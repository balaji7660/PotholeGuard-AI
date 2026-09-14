"""
models/transunet.py
Multi-Task TransUNet — the core perception backbone.

Architecture:
  1. CNN feature extractor (ResNet-50 via timm)
  2. Transformer encoder (ViT-style, applied on deepest CNN feature map)
  3. U-Net decoder with skip connections from CNN stages
  4. Three output heads:
       - Segmentation  → binary pothole mask
       - Depth         → per-pixel depth estimate
       - Uncertainty   → per-pixel perception uncertainty (MC-Dropout)

Returns
-------
dict:
    "segmentation" : (B, 1, H, W)   float32, values in [0,1]
    "depth"        : (B, 1, H, W)   float32, values in [min_depth, max_depth]
    "uncertainty"  : (B, 1, H, W)   float32, values in [0,1]
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.depth_head import DepthHead
from models.uncertainty_head import UncertaintyHead


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Project CNN feature map into 1D token sequence."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = 1) -> None:
        super().__init__()
        # patch_size=1 → each spatial location is one token (standard TransUNet)
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x : (B, C, H, W)
        returns : (B, N, hidden_size), H_grid, W_grid
        """
        x = self.proj(x)                  # (B, hidden_size, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single Pre-LN transformer block (attention + FFN)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn  = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        ffn_dim = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.drop(x) + residual
        # FFN
        residual = x
        x = self.ffn(self.norm2(x)) + residual
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks with learnable positional embedding."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, hidden_size, patch_size=1)
        # Positional embedding: allocated for max 32×32 tokens (1024 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x : (B, C, H, W) CNN feature map
        returns : (B, hidden_size, H, W)  — spatially restored
        """
        tokens, H, W = self.patch_embed(x)     # (B, N, D)
        N = tokens.shape[1]

        # Interpolate positional embedding if spatial dims differ from 32×32
        if N != self.pos_embed.shape[1]:
            pos = self.pos_embed.reshape(1, 32, 32, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(H, W), mode="bilinear", align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)  # (1, N, D)
        else:
            pos = self.pos_embed[:, :N, :]

        tokens = tokens + pos
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Reshape back to spatial map
        B, _, D = tokens.shape
        out = tokens.transpose(1, 2).reshape(B, D, H, W)  # (B, D, H, W)
        return out, H, W


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Single U-Net decoder stage: upsample + skip connection + conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Pad if spatial sizes differ (can happen with odd input dims)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Segmentation Head
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Final segmentation output head."""

    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.act  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


# ---------------------------------------------------------------------------
# TransUNet (full model)
# ---------------------------------------------------------------------------

class TransUNet(nn.Module):
    """
    Multi-Task TransUNet.

    Forward pass returns a dict:
        {
            "segmentation" : (B, 1, H, W)
            "depth"        : (B, 1, H, W)
            "uncertainty"  : (B, 1, H, W)
        }
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg    = cfg
        tu_cfg      = cfg.transunet
        trans_cfg   = tu_cfg.transformer
        dec_cfg     = tu_cfg.decoder

        # ---- CNN Backbone (ResNet-50 from timm) ----
        self.backbone = timm.create_model(
            tu_cfg.backbone.type,
            pretrained=tu_cfg.backbone.pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),   # stage outputs: 1/4, 1/8, 1/16, 1/32
        )
        if tu_cfg.backbone.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Determine backbone output channels via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, tu_cfg.image_size, tu_cfg.image_size)
            feats = self.backbone(dummy)
        self._feat_channels: List[int] = [f.shape[1] for f in feats]

        # ---- Transformer Encoder (on deepest stage) ----
        deepest_ch = self._feat_channels[-1]
        self.transformer = TransformerEncoder(
            in_channels=deepest_ch,
            hidden_size=trans_cfg.hidden_size,
            num_layers=trans_cfg.num_layers,
            num_heads=trans_cfg.num_heads,
            mlp_ratio=trans_cfg.mlp_ratio,
            dropout=trans_cfg.dropout,
            attn_dropout=trans_cfg.attention_dropout,
        )

        # ---- U-Net Decoder ----
        skip_chs  = list(reversed(self._feat_channels[:-1]))   # [256, 128, 64]
        up_chs    = dec_cfg.up_channels                         # [512, 256, 128, 64]

        self.decoder_blocks = nn.ModuleList()
        in_ch = trans_cfg.hidden_size
        for i, (skip_ch, out_ch) in enumerate(zip(skip_chs, up_chs)):
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # Final upsample (×2) to match input resolution, no skip
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        final_ch = in_ch // 2

        # ---- Output heads ----
        seg_cfg   = tu_cfg.segmentation_head
        depth_cfg = tu_cfg.depth_head
        unc_cfg   = tu_cfg.uncertainty_head

        self.seg_head  = SegmentationHead(final_ch, seg_cfg.num_classes)
        self.depth_head = DepthHead(
            in_channels=final_ch,
            out_channels=depth_cfg.out_channels,
            min_depth=depth_cfg.min_depth,
            max_depth=depth_cfg.max_depth,
        )
        self.unc_head = UncertaintyHead(
            in_channels=final_ch,
            out_channels=unc_cfg.out_channels,
            dropout_rate=unc_cfg.dropout_rate,
            mc_samples=unc_cfg.mc_samples,
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  normalised RGB tensor

        Returns
        -------
        dict with "segmentation", "depth", "uncertainty"
        """
        B, _, H, W = x.shape

        # 1. CNN feature extraction
        feats = self.backbone(x)  # list of 4 tensors at 1/4, 1/8, 1/16, 1/32

        # 2. Transformer on deepest feature map
        enc_out, _, _ = self.transformer(feats[-1])  # (B, D, H/32, W/32)

        # 3. U-Net decoder with skip connections
        dec = enc_out
        skip_feats = list(reversed(feats[:-1]))  # [1/16, 1/8, 1/4]
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_feats[i] if i < len(skip_feats) else None
            dec = block(dec, skip)

        # 4. Final upsample to full resolution
        dec = self.final_up(dec)
        # Ensure spatial size matches input
        if dec.shape[2:] != (H, W):
            dec = F.interpolate(dec, size=(H, W), mode="bilinear", align_corners=False)

        # 5. Output heads
        seg = self.seg_head(dec)    # (B, 1, H, W)
        dep = self.depth_head(dec)  # (B, 1, H, W)
        unc = self.unc_head(dec)    # (B, 1, H, W)

        return {
            "segmentation": seg,
            "depth":        dep,
            "uncertainty":  unc,
        }

    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run MC-Dropout inference: enable dropout at test time, run N forward
        passes, and return mean + variance for the uncertainty estimate.
        """
        self.unc_head.enable_mc_dropout()
        self.train()  # Enable dropout layers

        n_samples = self.cfg.transunet.uncertainty_head.mc_samples
        seg_samples, dep_samples, unc_samples = [], [], []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x)
                seg_samples.append(out["segmentation"])
                dep_samples.append(out["depth"])
                unc_samples.append(out["uncertainty"])

        self.eval()
        self.unc_head.disable_mc_dropout()

        seg_mean  = torch.stack(seg_samples).mean(0)
        dep_mean  = torch.stack(dep_samples).mean(0)
        unc_mean  = torch.stack(unc_samples).mean(0)
        seg_var   = torch.stack(seg_samples).var(0)   # Epistemic uncertainty proxy

        return {
            "segmentation": seg_mean,
            "depth":        dep_mean,
            "uncertainty":  unc_mean,
            "seg_variance": seg_var,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_transunet(cfg: DictConfig) -> TransUNet:
    """Instantiate and return a TransUNet from config."""
    model = TransUNet(cfg.transunet if hasattr(cfg, "transunet") else cfg)
    return model
