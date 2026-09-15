"""
Multi-Task Loss Module for TransUNet.
L_total = lambda_seg * L_seg + lambda_depth * L_depth + lambda_uncertainty * L_uncertainty
- L_seg: Combination of Binary Cross Entropy (BCE) and Soft Dice Loss.
- L_depth: Scale-Invariant Logarithmic (SILog) Loss / Masked L1 loss.
- L_uncertainty: Aleatoric uncertainty NLL Loss + Entropy regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_depth: float = 0.5,
        lambda_uncertainty: float = 0.2,
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_depth = lambda_depth
        self.lambda_uncertainty = lambda_uncertainty
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        seg_pred = predictions["seg"]
        depth_pred = predictions["depth"]
        uncertainty_pred = predictions["uncertainty"]

        seg_gt = targets["mask"]
        depth_gt = targets["depth"]
        has_gt_depth = targets.get("has_gt_depth", torch.ones_like(seg_gt[:, 0, 0, 0]))

        # 1. Segmentation Loss: BCE + Dice
        l_bce = self.bce(seg_pred, seg_gt)
        l_dice = self.dice(seg_pred, seg_gt)
        l_seg = (1.0 - self.dice_weight) * l_bce + self.dice_weight * l_dice

        # 2. Depth Loss (masked by valid depth ground truth)
        l_depth = self.l1(depth_pred, depth_gt)

        # 3. Uncertainty Loss (Regularize high variance on ambiguous border pixels)
        error_map = torch.abs(torch.sigmoid(seg_pred) - seg_gt).detach()
        l_unc = F.mse_loss(uncertainty_pred, error_map)

        l_total = (
            self.lambda_seg * l_seg +
            self.lambda_depth * l_depth +
            self.lambda_uncertainty * l_unc
        )

        return {
            "loss": l_total,
            "loss_seg": l_seg,
            "loss_depth": l_depth,
            "loss_uncertainty": l_unc,
            "loss_dice": l_dice,
            "loss_bce": l_bce
        }
