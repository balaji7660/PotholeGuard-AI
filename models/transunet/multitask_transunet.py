"""
Multi-Task TransUNet Architecture.
Combines ResNet CNN encoder, Vision Transformer bottleneck, U-Net progressive decoder,
and three simultaneous task heads:
1. Pothole Segmentation
2. Relative Depth Estimation
3. Uncertainty Estimation
"""
import torch
import torch.nn as nn
from typing import Dict, Any

from .encoder import ResNetEncoder
from .transformer import TransformerBottleneck
from .decoder import UNetDecoder
from .segmentation_head import SegmentationHead
from .depth_head import DepthHead
from .uncertainty_head import UncertaintyHead

class MultiTaskTransUNet(nn.Module):
    def __init__(
        self,
        pretrained_encoder: bool = False,
        embed_dim: int = 768,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        mc_dropout_rate: float = 0.2
    ):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained_encoder)
        self.transformer = TransformerBottleneck(
            in_channels=1024,
            embed_dim=embed_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads
        )
        self.decoder = UNetDecoder()
        self.seg_head = SegmentationHead(in_channels=32, num_classes=1)
        self.depth_head = DepthHead(in_channels=32)
        self.uncertainty_head = UncertaintyHead(in_channels=32, mc_dropout_rate=mc_dropout_rate)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor [B, 3, H, W]
            mc_dropout: Whether to activate MC dropout in uncertainty estimation
        Returns:
            dict with:
                'seg': Segmentation logits [B, 1, H, W]
                'depth': Relative depth map [B, 1, H, W] in [0, 1]
                'uncertainty': Uncertainty map [B, 1, H, W] in [0, 1]
        """
        # Encoder
        x_deep, skips = self.encoder(x)
        
        # Transformer bottleneck
        x_trans = self.transformer(x_deep)
        
        # Decoder
        shared_feat = self.decoder(x_trans, skips)
        
        # Heads
        seg_logits = self.seg_head(shared_feat)
        depth_map = self.depth_head(shared_feat)
        uncertainty_map = self.uncertainty_head(shared_feat, enable_mc_dropout=mc_dropout)
        
        return {
            'seg': seg_logits,
            'depth': depth_map,
            'uncertainty': uncertainty_map
        }
