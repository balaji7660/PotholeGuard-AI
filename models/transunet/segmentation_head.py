"""
Segmentation Head for Multi-Task TransUNet.
Outputs per-pixel pothole segmentation logits [B, 1, H, W].
"""
import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 32, num_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
