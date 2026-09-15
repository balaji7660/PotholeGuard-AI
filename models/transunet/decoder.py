"""
Decoder module for Multi-Task TransUNet.
Progressive upsampling decoder with skip-connections from encoder stages.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        # Upsample 2x
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDecoder(nn.Module):
    """
    U-Net progressive decoder taking bottleneck features (512 ch)
    and skip connections [x0 (64ch, H/2), x1 (256ch, H/4), x2 (512ch, H/8)].
    """
    def __init__(self):
        super().__init__()
        # Bottleneck: H/16 -> H/8, concat with x2 (512)
        self.dec1 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        # H/8 -> H/4, concat with x1 (256)
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        # H/4 -> H/2, concat with x0 (64)
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # Final upsample H/2 -> H/1
        self.dec4 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x0, x1, x2 = skips
        d1 = self.dec1(bottleneck, x2)
        d2 = self.dec2(d1, x1)
        d3 = self.dec3(d2, x0)
        d4 = self.dec4(d3, None)
        return d4
