"""
Depth Estimation Head for Multi-Task TransUNet.
Outputs relative per-pixel depth map [B, 1, H, W] in range [0, 1] using Sigmoid.
"""
import torch
import torch.nn as nn

class DepthHead(nn.Module):
    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
