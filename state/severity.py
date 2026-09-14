"""
state/severity.py
Pothole severity scorer.

Severity = weighted combination of:
  - Area (normalised to [0,1])
  - Average depth (normalised)
  - Minimum depth (normalised)
  - Lane-relative displacement (normalised)

NOTE: The specific formula and all coefficients below are an IMPLEMENTATION
CHOICE. The research paper describes severity as depending on these physical
features but does NOT specify exact coefficients. All weights are loaded from
configs/severity_config.yaml and must be tuned experimentally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig


@dataclass
class PotholeRegion:
    """Describes a single detected pothole region."""
    area_pixels:      float   # pixel area of connected component
    avg_depth:        float   # mean depth (metres / proxy units) inside region
    min_depth:        float   # minimum depth inside region
    lane_displacement: float  # signed offset from lane centre (metres)


class PotholeSeverityScorer:
    """
    Computes a normalised severity score in [0, 1] for a pothole region.

    Parameters
    ----------
    severity_cfg : DictConfig
        Loaded from configs/severity_config.yaml → severity section.
    """

    def __init__(self, severity_cfg: DictConfig) -> None:
        cfg = severity_cfg.severity
        self.w_area     = float(cfg.area_weight)
        self.w_avg_dep  = float(cfg.avg_depth_weight)
        self.w_min_dep  = float(cfg.min_depth_weight)
        self.w_disp     = float(cfg.lane_displacement_weight)

        self.max_area   = float(cfg.max_area_pixels)
        self.max_depth  = float(cfg.max_depth_metres)
        self.max_disp   = float(cfg.max_lane_displacement)

        self.clip_min   = float(cfg.clip_min)
        self.clip_max   = float(cfg.clip_max)

    # ------------------------------------------------------------------

    def score(self, region: PotholeRegion) -> float:
        """
        Compute severity score for a single pothole region.

        Parameters
        ----------
        region : PotholeRegion

        Returns
        -------
        severity : float in [0, 1]
        """
        # Normalise each component to [0, 1]
        f_area  = np.clip(region.area_pixels / (self.max_area + 1e-6), 0.0, 1.0)
        f_avg   = np.clip(region.avg_depth   / (self.max_depth + 1e-6), 0.0, 1.0)
        f_min   = np.clip(region.min_depth   / (self.max_depth + 1e-6), 0.0, 1.0)
        f_disp  = np.clip(abs(region.lane_displacement) / (self.max_disp + 1e-6), 0.0, 1.0)

        severity = (
            self.w_area    * f_area
            + self.w_avg_dep * f_avg
            + self.w_min_dep * f_min
            + self.w_disp    * f_disp
        )
        return float(np.clip(severity, self.clip_min, self.clip_max))

    def score_all(self, regions: list[PotholeRegion]) -> list[float]:
        """Score a list of regions. Returns empty list if no regions."""
        return [self.score(r) for r in regions]

    def max_severity(self, regions: list[PotholeRegion]) -> float:
        """Return the highest severity across all regions, or 0.0 if none."""
        scores = self.score_all(regions)
        return max(scores) if scores else 0.0
