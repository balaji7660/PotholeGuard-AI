"""
models/uncertainty_head.py
Perception uncertainty estimation head using MC-Dropout.

At training time: standard dropout is applied, head outputs a single
uncertainty proxy value.

At test time: call enable_mc_dropout() to keep dropout active across
multiple forward passes; the variance of those passes is the epistemic
uncertainty estimate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AlwaysDropout(nn.Module):
    """Dropout that stays active even in eval mode (for MC-Dropout)."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


class UncertaintyHead(nn.Module):
    """
    Uncertainty estimation head (MC-Dropout approach).

    Parameters
    ----------
    in_channels : int
        Decoder feature channels.
    out_channels : int
        1 for single-channel uncertainty map.
    dropout_rate : float
        Dropout probability used during MC sampling.
    mc_samples : int
        Number of MC forward passes (stored for reference; sampling is
        handled externally by TransUNet.predict_with_uncertainty).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        dropout_rate: float = 0.1,
        mc_samples: int = 10,
    ) -> None:
        super().__init__()
        self.mc_samples   = mc_samples
        self.dropout_rate = dropout_rate
        self._mc_active   = False

        self.std_dropout = nn.Dropout2d(p=dropout_rate)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
        )
        self.act = nn.Sigmoid()

    # ------------------------------------------------------------------
    # MC-Dropout control
    # ------------------------------------------------------------------

    def enable_mc_dropout(self) -> None:
        """Switch to always-active dropout for MC inference."""
        self._mc_active = True

    def disable_mc_dropout(self) -> None:
        """Restore normal dropout behaviour."""
        self._mc_active = False

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        uncertainty : (B, 1, H, W)  float32, values in [0, 1]
        """
        if self._mc_active:
            # During MC-Dropout: apply dropout unconditionally
            x = F.dropout2d(x, p=self.dropout_rate, training=True)
        else:
            x = self.std_dropout(x)

        return self.act(self.conv(x))
