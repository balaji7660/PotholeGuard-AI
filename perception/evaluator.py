"""
perception/evaluator.py
Computes per-batch and aggregate evaluation metrics for the perception model.

Metrics:
  - Dice coefficient (F1 score on binary segmentation)
  - IoU (Intersection over Union / Jaccard index)
  - Depth RMSE (Root Mean Squared Error, in dataset depth units)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


class PerceptionEvaluator:
    """
    Stateless evaluator: call compute_batch per batch and aggregate manually,
    or use accumulate() / summarise() for a streaming interface.
    """

    def __init__(self, seg_threshold: float = 0.5) -> None:
        self.seg_threshold = seg_threshold
        self._reset()

    # ------------------------------------------------------------------
    # Stateless per-batch
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        pred_seg: torch.Tensor,
        gt_seg:   torch.Tensor,
        pred_dep: Optional[torch.Tensor] = None,
        gt_dep:   Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        pred_seg  : (B, 1, H, W) float32 [0,1]
        gt_seg    : (B, 1, H, W) float32 {0,1}
        pred_dep  : (B, 1, H, W) float32 [optional]
        gt_dep    : (B, 1, H, W) float32 [optional]
        valid_mask: (B, 1, H, W) bool    [optional]

        Returns
        -------
        dict of scalar metric values.
        """
        pred_bin = (pred_seg >= self.seg_threshold).float()
        metrics: Dict[str, float] = {}

        # Dice
        metrics["dice"] = self._dice(pred_bin, gt_seg)

        # IoU
        metrics["iou"] = self._iou(pred_bin, gt_seg)

        # Depth RMSE
        if pred_dep is not None and gt_dep is not None:
            metrics["depth_rmse"] = self._rmse(pred_dep, gt_dep, valid_mask)

        return metrics

    # ------------------------------------------------------------------
    # Streaming accumulator
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._sum: Dict[str, float] = {}
        self._count: int = 0

    def accumulate(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sum[k] = self._sum.get(k, 0.0) + v
        self._count += 1

    def summarise(self) -> Dict[str, float]:
        if self._count == 0:
            return {}
        result = {k: v / self._count for k, v in self._sum.items()}
        self._reset()
        return result

    # ------------------------------------------------------------------
    # Internal metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum().item()
        return (2.0 * inter + smooth) / (pred.sum().item() + target.sum().item() + smooth)

    @staticmethod
    def _iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum().item()
        union  = (pred + target - pred * target).sum().item()
        return (inter + smooth) / (union + smooth)

    @staticmethod
    def _rmse(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> float:
        if valid_mask is not None:
            pred   = pred[valid_mask]
            target = target[valid_mask]
        if pred.numel() == 0:
            return 0.0
        return torch.sqrt(((pred - target) ** 2).mean()).item()
