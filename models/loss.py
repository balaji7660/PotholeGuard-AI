"""
models/loss.py
Multi-task loss function combining:
  - Segmentation loss  (BCE + Dice)
  - Depth loss         (Scale-Invariant Log / L1 / Smooth-L1)
  - Uncertainty loss   (regularisation: entropy maximisation proxy)

All weights are loaded from configs/training_config.yaml.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Component losses
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : (B, 1, H, W)  float32, pred in [0,1], target in {0,1}.
        """
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class SILogLoss(nn.Module):
    """
    Scale-Invariant Log (SILog) depth loss.
    λ is the variance-weighting term (Eigen et al., 2014).
    Implementation choice — λ loaded from config.
    """

    def __init__(self, lam: float = 0.85) -> None:
        super().__init__()
        self.lam = lam

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred, target : (B, 1, H, W)  positive float32 depth values.
        valid_mask   : (B, 1, H, W)  bool — True where depth is valid.
        """
        eps = 1e-6
        if valid_mask is not None:
            pred   = pred[valid_mask]
            target = target[valid_mask]

        log_diff = torch.log(pred.clamp(min=eps)) - torch.log(target.clamp(min=eps))
        loss = (log_diff ** 2).mean() - self.lam * (log_diff.mean() ** 2)
        return loss.clamp(min=0.0)


class UncertaintyRegLoss(nn.Module):
    """
    Uncertainty regularisation loss.

    Encourages the uncertainty head to produce meaningful (not collapsed) output
    by maximising entropy of the uncertainty distribution — implementation choice.
    Entropy of a Bernoulli: H = -(u log u + (1-u) log(1-u))
    Maximising H ↔ minimising -H → this term penalises collapsed (near-0 / near-1) outputs.
    """

    def forward(self, uncertainty: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        u = uncertainty.clamp(eps, 1.0 - eps)
        entropy = -(u * torch.log(u) + (1.0 - u) * torch.log(1.0 - u))
        # Negate: we want to maximise entropy, so the *loss* = -entropy
        return -entropy.mean()


# ---------------------------------------------------------------------------
# Multi-Task Loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for the TransUNet.

    L_total = w_seg  * L_seg
            + w_depth * L_depth
            + w_unc   * L_uncertainty

    All weights are loaded from configs/training_config.yaml.
    """

    def __init__(self, loss_cfg: DictConfig) -> None:
        super().__init__()
        self.w_seg   = float(loss_cfg.segmentation_weight)
        self.w_depth = float(loss_cfg.depth_weight)
        self.w_unc   = float(loss_cfg.uncertainty_weight)

        bce_w  = float(loss_cfg.seg_bce_weight)
        dice_w = float(loss_cfg.seg_dice_weight)

        # Segmentation: BCE + Dice combo
        self.bce_loss  = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.bce_w     = bce_w
        self.dice_w    = dice_w

        # Depth
        depth_type = loss_cfg.depth_loss_type
        if depth_type == "silog":
            self.depth_loss: nn.Module = SILogLoss(lam=float(loss_cfg.silog_lambda))
        elif depth_type == "l1":
            self.depth_loss = nn.L1Loss()
        elif depth_type == "smooth_l1":
            self.depth_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown depth loss type: {depth_type}")

        # Uncertainty regularisation
        self.unc_loss = UncertaintyRegLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        outputs : dict
            "segmentation" : (B, 1, H, W)
            "depth"        : (B, 1, H, W)
            "uncertainty"  : (B, 1, H, W)

        targets : dict
            "mask"  : (B, 1, H, W)   binary segmentation ground truth
            "depth" : (B, 1, H, W)   depth ground truth (may be pseudo)
            "valid_depth_mask" : (B, 1, H, W) bool  [optional]

        Returns
        -------
        dict with keys:
            "total", "seg", "depth", "uncertainty"
        """
        pred_seg  = outputs["segmentation"]
        pred_dep  = outputs["depth"]
        pred_unc  = outputs["uncertainty"]

        gt_seg    = targets["mask"]
        gt_dep    = targets["depth"]
        valid_mask = targets.get("valid_depth_mask", None)

        # Segmentation loss
        l_bce  = self.bce_loss(pred_seg, gt_seg)
        l_dice = self.dice_loss(pred_seg, gt_seg)
        l_seg  = self.bce_w * l_bce + self.dice_w * l_dice

        # Depth loss
        if isinstance(self.depth_loss, SILogLoss):
            l_dep = self.depth_loss(pred_dep, gt_dep, valid_mask)
        else:
            if valid_mask is not None:
                l_dep = self.depth_loss(pred_dep[valid_mask], gt_dep[valid_mask])
            else:
                l_dep = self.depth_loss(pred_dep, gt_dep)

        # Uncertainty regularisation
        l_unc = self.unc_loss(pred_unc)

        l_total = self.w_seg * l_seg + self.w_depth * l_dep + self.w_unc * l_unc

        return {
            "total":       l_total,
            "seg":         l_seg,
            "depth":       l_dep,
            "uncertainty": l_unc,
        }
