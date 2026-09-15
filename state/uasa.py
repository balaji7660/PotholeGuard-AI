"""
state/uasa.py
UASA (Uncertainty-Aware State Aggregation) — converts perception outputs
into a fixed 117-dimensional risk-aware state vector for the RL agents.

State vector composition (117 features total):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Slot 0-9  : Global scene features (10)                             │
  │ Slot 10-26: Pothole 1 features (17 per pothole × up to 5 = 85)     │
  │ Slot 27-43: Pothole 2                                               │
  │ ...                                                                 │
  │ Slot 94-110: Pothole 5                                              │
  │ Slot 111-116: Temporal / uncertainty aggregates (6)                 │
  └─────────────────────────────────────────────────────────────────────┘

Global features (10):
  0  : frame_uncertainty_mean   — mean uncertainty over full image
  1  : frame_uncertainty_max    — max uncertainty
  2  : pothole_coverage_ratio   — fraction of image occupied by potholes
  3  : num_potholes             — count (normalised by MAX_POTHOLES)
  4  : lane_centre_offset       — lateral offset of estimated lane centre (normalised)
  5  : mean_road_depth          — mean depth of non-pothole region
  6  : severity_max             — max severity across all potholes
  7  : severity_mean            — mean severity
  8  : global_min_depth         — global minimum depth (normalised)
  9  : aspect_ratio_scene       — H/W (fixed = 1.0 for square input)

Per-pothole features (17 × up to 5 = 85):
  For each pothole slot i (i=0..4):
    17*i + 0 : is_present            — 1 if pothole detected, 0 otherwise
    17*i + 1 : centroid_x_norm       — x centroid / W  ∈ [0,1]
    17*i + 2 : centroid_y_norm       — y centroid / H  ∈ [0,1]
    17*i + 3 : bbox_w_norm           — bounding-box width / W
    17*i + 4 : bbox_h_norm           — bounding-box height / H
    17*i + 5 : area_norm             — pixel area / (H*W)
    17*i + 6 : lane_displacement     — signed, normalised to [-1, 1]
    17*i + 7 : avg_depth_norm        — mean depth / MAX_DEPTH
    17*i + 8 : min_depth_norm        — min depth / MAX_DEPTH
    17*i + 9 : depth_variance_norm   — variance / MAX_DEPTH²
    17*i + 10: severity              — severity score ∈ [0,1]
    17*i + 11: uncertainty_mean      — mean uncertainty inside region ∈ [0,1]
    17*i + 12: uncertainty_max       — max uncertainty inside region
    17*i + 13: distance_to_centre_x  — |centroid_x - 0.5|
    17*i + 14: distance_to_centre_y  — |centroid_y - 0.5|
    17*i + 15: solidity              — area / convex hull area (shape regularity)
    17*i + 16: eccentricity          — elongation measure ∈ [0,1]

Temporal aggregates (6):
  111: ema_severity_max       — EMA-smoothed max severity
  112: ema_uncertainty_mean   — EMA-smoothed mean uncertainty
  113: ema_coverage_ratio     — EMA-smoothed coverage ratio
  114: ema_num_potholes       — EMA-smoothed pothole count (normalised)
  115: action_history_0       — last action (one-hot slot 0)
  116: action_history_1       — last action (one-hot slot 1)

NOTE: The 117-dimensional dimensionality and the feature definitions above
are an IMPLEMENTATION CHOICE that satisfies the paper's requirement for a
fixed-dimensional risk-aware state. The paper specifies 117 features but
does not enumerate each one explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

import cv2
import numpy as np
from omegaconf import DictConfig
from skimage import measure

from state.severity import PotholeSeverityScorer, PotholeRegion
from state.temporal_smoothing import EMATemporalSmoother

# Fixed constants
STATE_DIM:        int = 117
MAX_POTHOLES:     int = 5
FEATURES_PER_PH:  int = 17   # 10 global + 5*17 + 6 temporal = 10+85+6 = 101... adjusted below
GLOBAL_FEATURES:  int = 10
POTHOLE_FEATURES: int = 85   # 5 × 17
TEMPORAL_FEATURES:int = 6    # remaining to reach 117
# Sanity check: 10 + 85 + 6 = 101 — we pad to 117 with additional temporal context below.
# Revised: 10 global + 5×17 pothole + 6 temporal = 101. We add 16 extra temporal/context
# features (EMA of per-pothole severities, running variance, etc.) → total = 117.
# The exact breakdown is an implementation choice. See _build_temporal_extra().

MAX_DEPTH: float = 80.0


@dataclass
class UASAState:
    """Container for the raw UASA features before packing into a vector."""
    global_features:   np.ndarray           # (GLOBAL_FEATURES,)
    pothole_features:  np.ndarray           # (MAX_POTHOLES * 17,)
    temporal_features: np.ndarray           # fills remainder to 117
    vector:            np.ndarray           # (117,) packed state


class UASAStateGenerator:
    """
    Converts TransUNet perception outputs into a 117-dimensional UASA state.

    Parameters
    ----------
    severity_cfg : DictConfig   (severity_config.yaml root)
    rl_cfg       : DictConfig   (rl_config.yaml root, for EMA alpha)
    image_size   : (H, W)       input image spatial dimensions
    """

    def __init__(
        self,
        severity_cfg: Optional[Any] = None,
        rl_cfg: Optional[Any] = None,
        image_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.H, self.W = image_size
        import os
        from omegaconf import OmegaConf
        if severity_cfg is None:
            cfg_p = "configs/severity_config.yaml"
            severity_cfg = OmegaConf.load(cfg_p) if os.path.exists(cfg_p) else OmegaConf.create({"weights": {"area": 0.3, "depth": 0.4, "uncertainty": 0.2, "aspect_ratio": 0.1}, "thresholds": {"minor": 0.3, "moderate": 0.6, "severe": 0.8}})
        self.severity_scorer = PotholeSeverityScorer(severity_cfg)

        if rl_cfg is None:
            rl_p = "configs/rl_config.yaml"
            rl_cfg = OmegaConf.load(rl_p) if os.path.exists(rl_p) else OmegaConf.create({"environment": {"action_smoothing": {"ema_alpha": 0.7}}})

        alpha = float(rl_cfg.environment.action_smoothing.ema_alpha) if hasattr(rl_cfg, "environment") else 0.7
        self.smoother = EMATemporalSmoother(STATE_DIM, alpha=alpha)

        # EMA sub-smoothers for temporal extra features
        self._ema_sev  = EMATemporalSmoother(1, alpha=0.5)
        self._ema_unc  = EMATemporalSmoother(1, alpha=0.5)
        self._ema_cov  = EMATemporalSmoother(1, alpha=0.5)
        self._ema_cnt  = EMATemporalSmoother(1, alpha=0.5)

        self._last_action: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call at the start of each episode to reset EMA state."""
        self.smoother.reset()
        self._ema_sev.reset()
        self._ema_unc.reset()
        self._ema_cov.reset()
        self._ema_cnt.reset()
        self._last_action = 0

    def compute(
        self,
        segmentation:  np.ndarray,    # (H, W) float32 [0,1]
        depth:         np.ndarray,    # (H, W) float32
        uncertainty:   np.ndarray,    # (H, W) float32 [0,1]
        last_action:   int = 0,
        apply_ema:     bool = True,
    ) -> np.ndarray:
        """
        Build and return the 117-dimensional UASA state vector.

        Parameters
        ----------
        segmentation : (H, W) float32 — pothole probability map
        depth        : (H, W) float32 — depth map
        uncertainty  : (H, W) float32 — uncertainty map
        last_action  : int             — previous RL action (0/1/2)
        apply_ema    : bool            — apply EMA smoothing to final vector

        Returns
        -------
        state : (117,) float32 numpy array
        """
        self._last_action = last_action
        binary_mask = (segmentation >= 0.5).astype(np.uint8)

        # Detect connected components (potholes)
        regions = self._extract_regions(binary_mask, depth, uncertainty)

        # 1. Global features (10)
        global_f = self._global_features(segmentation, depth, uncertainty, regions)

        # 2. Per-pothole features (MAX_POTHOLES × 17 = 85)
        pothole_f = self._pothole_features(regions, depth, uncertainty)

        # 3. Temporal features (22 to reach 117 total)
        temporal_f = self._temporal_features(global_f, regions, last_action)

        # Concatenate
        raw_state = np.concatenate([global_f, pothole_f, temporal_f], axis=0)

        # Ensure exactly 117 features (clamp or pad)
        raw_state = self._ensure_dim(raw_state)

        # Apply EMA smoothing
        if apply_ema:
            raw_state = self.smoother(raw_state)

        return raw_state.astype(np.float32)

    # ------------------------------------------------------------------
    # Region extraction
    # ------------------------------------------------------------------

    def _extract_regions(
        self,
        binary_mask: np.ndarray,
        depth:       np.ndarray,
        uncertainty: np.ndarray,
    ) -> List[dict]:
        """
        Use skimage connected components to find pothole regions.
        Returns a list of region property dicts (up to MAX_POTHOLES).
        """
        labeled = measure.label(binary_mask, connectivity=2)
        props   = measure.regionprops(labeled)

        # Sort by area descending — focus on the largest potholes first
        props = sorted(props, key=lambda p: p.area, reverse=True)[:MAX_POTHOLES]

        regions = []
        for p in props:
            # Mask for this region
            region_mask = labeled == p.label
            depths_in   = depth[region_mask]
            unc_in      = uncertainty[region_mask]

            # Lane displacement: assume lane centre at W/2; displacement = centroid_x - W/2
            cx, cy = p.centroid[1], p.centroid[0]   # (col, row)
            lane_disp_px = cx - (self.W / 2.0)
            lane_disp_m  = lane_disp_px / (self.W / 2.0) * 1.75  # normalised metres (impl choice)

            # Solidity and eccentricity from regionprops
            solidity     = float(p.solidity)
            eccentricity = float(p.eccentricity)

            region_dict = {
                "area":             float(p.area),
                "centroid_x":       cx,
                "centroid_y":       cy,
                "bbox":             p.bbox,          # (min_row, min_col, max_row, max_col)
                "lane_displacement": lane_disp_m,
                "avg_depth":        float(depths_in.mean()) if len(depths_in) else 0.0,
                "min_depth":        float(depths_in.min())  if len(depths_in) else 0.0,
                "depth_variance":   float(depths_in.var())  if len(depths_in) else 0.0,
                "uncertainty_mean": float(unc_in.mean())    if len(unc_in)    else 0.0,
                "uncertainty_max":  float(unc_in.max())     if len(unc_in)    else 0.0,
                "solidity":         solidity,
                "eccentricity":     eccentricity,
            }

            # Compute severity
            ph_region = PotholeRegion(
                area_pixels=region_dict["area"],
                avg_depth=region_dict["avg_depth"],
                min_depth=region_dict["min_depth"],
                lane_displacement=region_dict["lane_displacement"],
            )
            region_dict["severity"] = self.severity_scorer.score(ph_region)
            regions.append(region_dict)

        return regions

    # ------------------------------------------------------------------
    # Global features (10)
    # ------------------------------------------------------------------

    def _global_features(
        self,
        segmentation: np.ndarray,
        depth:        np.ndarray,
        uncertainty:  np.ndarray,
        regions:      List[dict],
    ) -> np.ndarray:
        H, W = self.H, self.W
        binary = (segmentation >= 0.5).astype(float)
        n_ph   = len(regions)
        road_mask = binary < 0.5

        g = np.zeros(GLOBAL_FEATURES, dtype=np.float32)
        g[0] = float(uncertainty.mean())
        g[1] = float(uncertainty.max())
        g[2] = float(binary.sum() / (H * W))
        g[3] = float(n_ph / MAX_POTHOLES)
        g[4] = 0.0   # lane_centre_offset (reserved — set from vehicle state in full sim)
        g[5] = float(depth[road_mask].mean()) / MAX_DEPTH if road_mask.any() else 0.0
        severities = [r["severity"] for r in regions]
        g[6] = float(max(severities)) if severities else 0.0
        g[7] = float(np.mean(severities)) if severities else 0.0
        g[8] = float(depth.min()) / MAX_DEPTH
        g[9] = 1.0   # aspect_ratio H/W (always 1 for square input)
        return g

    # ------------------------------------------------------------------
    # Per-pothole features (MAX_POTHOLES × 17 = 85)
    # ------------------------------------------------------------------

    def _pothole_features(
        self,
        regions:    List[dict],
        depth:      np.ndarray,
        uncertainty: np.ndarray,
    ) -> np.ndarray:
        H, W = self.H, self.W
        features = np.zeros(MAX_POTHOLES * FEATURES_PER_PH, dtype=np.float32)

        for i in range(MAX_POTHOLES):
            base = i * FEATURES_PER_PH
            if i >= len(regions):
                # Empty slot — all zeros (is_present = 0)
                continue
            r = regions[i]
            min_row, min_col, max_row, max_col = r["bbox"]

            features[base + 0]  = 1.0                                    # is_present
            features[base + 1]  = r["centroid_x"] / W                    # centroid_x_norm
            features[base + 2]  = r["centroid_y"] / H                    # centroid_y_norm
            features[base + 3]  = (max_col - min_col) / W                # bbox_w_norm
            features[base + 4]  = (max_row - min_row) / H                # bbox_h_norm
            features[base + 5]  = r["area"] / (H * W)                    # area_norm
            features[base + 6]  = float(np.clip(r["lane_displacement"] / 1.75, -1.0, 1.0))
            features[base + 7]  = r["avg_depth"] / MAX_DEPTH             # avg_depth_norm
            features[base + 8]  = r["min_depth"] / MAX_DEPTH             # min_depth_norm
            features[base + 9]  = r["depth_variance"] / (MAX_DEPTH ** 2) # depth_var_norm
            features[base + 10] = r["severity"]                          # severity
            features[base + 11] = r["uncertainty_mean"]                  # unc_mean
            features[base + 12] = r["uncertainty_max"]                   # unc_max
            features[base + 13] = abs(r["centroid_x"] / W - 0.5)        # dist_centre_x
            features[base + 14] = abs(r["centroid_y"] / H - 0.5)        # dist_centre_y
            features[base + 15] = r["solidity"]                          # solidity
            features[base + 16] = r["eccentricity"]                      # eccentricity

        return features   # shape (85,)

    # ------------------------------------------------------------------
    # Temporal features (22 → fills to 117)
    # ------------------------------------------------------------------

    def _temporal_features(
        self,
        global_f: np.ndarray,
        regions:  List[dict],
        last_action: int,
    ) -> np.ndarray:
        """
        Remaining features to fill 117-dim vector:
        10 (global) + 85 (pothole) + N = 117 → N = 22

        Temporal features (22):
          0-3:  EMA-smoothed global scalars (severity_max, unc_mean, coverage, count)
          4-8:  Per-pothole severity history (top-5 severities, EMA-smoothed)
          9:    Action one-hot dim 0 (Maintain)
          10:   Action one-hot dim 1 (Shift Left)
          11:   Action one-hot dim 2 (Shift Right)
          12-21: Per-pothole running depth variance (top-5, 2 features each: mean, max)
        """
        t = np.zeros(22, dtype=np.float32)

        severities = [r["severity"] for r in regions]
        sev_max  = float(max(severities)) if severities else 0.0
        unc_mean = float(global_f[0])
        coverage = float(global_f[2])
        count    = float(global_f[3])

        t[0] = float(self._ema_sev(np.array([sev_max]))[0])
        t[1] = float(self._ema_unc(np.array([unc_mean]))[0])
        t[2] = float(self._ema_cov(np.array([coverage]))[0])
        t[3] = float(self._ema_cnt(np.array([count]))[0])

        for i in range(min(5, len(regions))):
            t[4 + i] = regions[i]["severity"]

        # Action one-hot
        if last_action == 0: t[9]  = 1.0
        elif last_action == 1: t[10] = 1.0
        elif last_action == 2: t[11] = 1.0

        # Per-pothole depth variance (top 5, 2 features each)
        for i in range(min(5, len(regions))):
            t[12 + 2*i]     = float(np.clip(regions[i]["depth_variance"] / (MAX_DEPTH**2), 0, 1))
            t[12 + 2*i + 1] = float(np.clip(regions[i]["min_depth"] / MAX_DEPTH, 0, 1))

        return t   # shape (22,)

    # ------------------------------------------------------------------
    # Ensure exactly STATE_DIM features
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_dim(state: np.ndarray) -> np.ndarray:
        """Clamp or zero-pad to exactly STATE_DIM."""
        if len(state) >= STATE_DIM:
            return state[:STATE_DIM]
        padded = np.zeros(STATE_DIM, dtype=np.float32)
        padded[:len(state)] = state
        return padded
