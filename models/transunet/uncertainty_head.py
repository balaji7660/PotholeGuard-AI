"""
Uncertainty Estimation Head for Multi-Task TransUNet.
Provides epistemic / aleatoric uncertainty map [B, 1, H, W] normalized between [0, 1].
Supports active Monte Carlo Dropout at inference time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyHead(nn.Module):
    def __init__(self, in_channels: int = 32, mc_dropout_rate: float = 0.2):
        super().__init__()
        self.mc_dropout_rate = mc_dropout_rate
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=mc_dropout_rate),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, enable_mc_dropout: bool = False) -> torch.Tensor:
        if enable_mc_dropout:
            # Force dropout active even during eval mode
            for m in self.head.modules():
                if isinstance(m, nn.Dropout2d) or isinstance(m, nn.Dropout):
                    m.train()
        return self.head(x)
