"""
Encoder module for Multi-Task TransUNet.
Provides standard CNN feature extraction (e.g. ResNet-50 style or lightweight Conv backbone)
with intermediate feature extraction for U-Net skip connections.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResNetEncoder(nn.Module):
    """
    ResNet-50 based feature extractor outputting skip connections at:
    - stem (skip1): 64 channels, 1/2 resolution
    - layer1 (skip2): 256 channels, 1/4 resolution
    - layer2 (skip3): 512 channels, 1/8 resolution
    - layer3 (skip4): 1024 channels, 1/16 resolution
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Initialize ResNet-50 weights or random init
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 ch
        self.layer2 = resnet.layer2  # 512 ch
        self.layer3 = resnet.layer3  # 1024 ch

    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)         # [B, 64, H/2, W/2]
        x_pool = self.maxpool(x0) # [B, 64, H/4, W/4]
        x1 = self.layer1(x_pool)  # [B, 256, H/4, W/4]
        x2 = self.layer2(x1)      # [B, 512, H/8, W/8]
        x3 = self.layer3(x2)      # [B, 1024, H/16, W/16]
        return x3, [x0, x1, x2]
