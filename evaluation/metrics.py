"""
evaluation/metrics.py
Evaluation metrics for the full pipeline.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import time


class PipelineMetrics:
    """Accumulates per-frame metrics across an evaluation run."""

    def __init__(self) -> None:
        self._dice:  List[float] = []
        self._iou:   List[float] = []
        self._depth_rmse: List[float] = []
        self._lat_dev:    List[float] = []
        self._collisions: List[bool]  = []
        self._srl_overrides: List[bool] = []
        self._fps_times:  List[float] = []
        self._actions:    List[int]   = []

    def add(
        self,
        dice: Optional[float] = None,
        iou: Optional[float] = None,
        depth_rmse: Optional[float] = None,
        lateral_deviation: Optional[float] = None,
        collision: Optional[bool] = None,
        srl_override: Optional[bool] = None,
        fps: Optional[float] = None,
        action: Optional[int] = None,
    ) -> None:
        if dice         is not None: self._dice.append(dice)
        if iou          is not None: self._iou.append(iou)
        if depth_rmse   is not None: self._depth_rmse.append(depth_rmse)
        if lateral_deviation is not None: self._lat_dev.append(lateral_deviation)
        if collision    is not None: self._collisions.append(collision)
        if srl_override is not None: self._srl_overrides.append(srl_override)
        if fps          is not None: self._fps_times.append(fps)
        if action       is not None: self._actions.append(action)

    def summary(self) -> Dict[str, float]:
        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        return {
            "dice":              safe_mean(self._dice),
            "iou":               safe_mean(self._iou),
            "depth_rmse":        safe_mean(self._depth_rmse),
            "lateral_deviation": safe_mean(self._lat_dev),
            "collision_rate":    safe_mean([float(c) for c in self._collisions]),
            "srl_intervention":  safe_mean([float(s) for s in self._srl_overrides]),
            "fps":               safe_mean(self._fps_times),
            "n_frames":          float(len(self._dice) or len(self._actions)),
        }

    def reset(self) -> None:
        self.__init__()
