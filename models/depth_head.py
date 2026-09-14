"""
models/depth_head.py
Depth estimation head for the TransUNet.

Outputs per-pixel depth estimates scaled to [min_depth, max_depth].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """
    Depth estimation head.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels from the decoder.
    out_channels : int
        1 for single-channel depth map.
    min_depth : float
        Minimum depth value in metres (clips sigmoid output).
    max_depth : float
        Maximum depth value in metres.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
    ) -> None:
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        depth : (B, 1, H, W)  float32, values in [min_depth, max_depth]
        """
        raw   = self.act(self.conv(x))
        depth = self.min_depth + (self.max_depth - self.min_depth) * raw
        return depth
