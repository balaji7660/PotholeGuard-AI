"""
data/pseudo_depth.py
MiDaS v3 pseudo-depth generator for RGB-only dataset samples.

Loaded via torch.hub (no extra pip package needed beyond torch + timm).
The generated depth is a relative/affine-ambiguous depth map; it is useful
for the perception pipeline as a proxy but cannot be used as metric ground truth.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig


class PseudoDepthGenerator:
    """
    Wraps MiDaS v3 (via torch.hub) to produce single-image pseudo-depth maps.

    Parameters
    ----------
    cfg : DictConfig
        Full dataset configuration. Uses cfg.pseudo_depth sub-section.
    device : str | None
        Override device selection. If None, uses cfg.pseudo_depth.device.
    """

    # Model name → torch.hub repo / model_type mapping
    _HUB_REPO = "intel-isl/MiDaS"
    _MODEL_TYPES = {
        "DPT_Large":  "DPT_Large",
        "DPT_Hybrid": "DPT_Hybrid",
        "MiDaS_small": "MiDaS_small",
    }

    def __init__(self, cfg: DictConfig, device: Optional[str] = None) -> None:
        pd_cfg = cfg.pseudo_depth
        self.model_name: str = pd_cfg.model
        self.output_scale: float = float(pd_cfg.output_scale)
        self.invert: bool = bool(pd_cfg.invert)

        # Device resolution
        dev_str = device or pd_cfg.device
        if dev_str == "auto":
            dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev_str)

        self._model: Optional[torch.nn.Module] = None
        self._transform = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load MiDaS from torch.hub (cached after first call)."""
        model_type = self._MODEL_TYPES.get(self.model_name, self.model_name)
        try:
            self._model = torch.hub.load(
                self._HUB_REPO,
                model_type,
                pretrained=True,
                trust_repo=True,
            )
            self._model.to(self.device).eval()

            midas_transforms = torch.hub.load(
                self._HUB_REPO,
                "transforms",
                trust_repo=True,
            )
            if model_type in ("DPT_Large", "DPT_Hybrid"):
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

        except Exception as exc:
            warnings.warn(
                f"Failed to load MiDaS ({exc}). "
                "Returning zero depth maps. Install torch.hub dependencies "
                "or provide ground-truth depth maps.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Generate a pseudo-depth map for an RGB image.

        Parameters
        ----------
        image_rgb : np.ndarray  (H, W, 3) uint8

        Returns
        -------
        depth : np.ndarray  (H, W) float32  (relative depth in arbitrary units)
        """
        if self._model is None:
            self._load_model()

        if self._model is None:
            # Fallback: return zero depth
            h, w = image_rgb.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

        with torch.no_grad():
            input_batch = self._transform(image_rgb).to(self.device)
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # MiDaS outputs *inverse* depth — invert to get approximate metric depth
        if self.invert:
            eps = 1e-6
            depth = 1.0 / (depth + eps)

        depth = depth * self.output_scale
        return depth
