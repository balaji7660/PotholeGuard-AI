"""
perception/inference.py
End-to-end perception pipeline: raw RGB image → segmentation + depth + uncertainty.

Usage
-----
pipeline = PerceptionPipeline.from_config(model_cfg, ckpt_path, device)
result   = pipeline.infer(image_bgr)   # numpy BGR uint8
# result["segmentation"]  : np.ndarray (H, W) float32 [0,1]
# result["depth"]         : np.ndarray (H, W) float32
# result["uncertainty"]   : np.ndarray (H, W) float32 [0,1]
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from models.transunet import TransUNet


class PerceptionPipeline:
    """
    Wraps the TransUNet for production inference.

    Parameters
    ----------
    model       : TransUNet  (weights already loaded)
    device      : torch.device
    image_size  : (H, W)  — resize target before feeding the model
    mean, std   : ImageNet normalisation constants
    use_mc      : bool  — use MC-Dropout for uncertainty (slower but richer)
    """

    def __init__(
        self,
        model: TransUNet,
        device: torch.device,
        image_size: tuple[int, int] = (512, 512),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std:  tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_mc: bool = False,
    ) -> None:
        self.model      = model.to(device).eval()
        self.device     = device
        self.image_size = image_size
        self.mean       = np.array(mean, dtype=np.float32)
        self.std        = np.array(std,  dtype=np.float32)
        self.use_mc     = use_mc

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_cfg: DictConfig,
        ckpt_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_mc: bool = False,
    ) -> "PerceptionPipeline":
        """
        Build pipeline from config and optional checkpoint.

        Parameters
        ----------
        model_cfg : DictConfig  (model_config.yaml root or .transunet sub-key)
        ckpt_path : str | None  path to .pt checkpoint
        device    : torch.device | None  (auto-detect if None)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TransUNet(model_cfg)

        if ckpt_path and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("model_state", ckpt)
            model.load_state_dict(state_dict)

        img_size = tuple(model_cfg.transunet.image_size
                         if hasattr(model_cfg.transunet, "image_size")
                         else (512, 512))
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        return cls(model=model, device=device, image_size=img_size, use_mc=use_mc)

    # ------------------------------------------------------------------
    # Preprocessing / Postprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → normalised float tensor (1, 3, H, W)."""
        h, w = self.image_size
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device)

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """(1, 1, H, W) → (H, W) numpy float32."""
        return tensor.squeeze().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, image_bgr: np.ndarray) -> dict:
        """
        Run the full perception pipeline on a single BGR image.

        Parameters
        ----------
        image_bgr : np.ndarray  (H, W, 3) uint8 BGR

        Returns
        -------
        dict:
            "segmentation"  : (H, W) float32 [0,1]
            "depth"         : (H, W) float32
            "uncertainty"   : (H, W) float32 [0,1]
            "inference_ms"  : float  latency in milliseconds
        """
        t0 = time.perf_counter()
        tensor = self._preprocess(image_bgr)

        if self.use_mc:
            outputs = self.model.predict_with_uncertainty(tensor)
        else:
            self.model.eval()
            outputs = self.model(tensor)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "segmentation": self._to_numpy(outputs["segmentation"]),
            "depth":        self._to_numpy(outputs["depth"]),
            "uncertainty":  self._to_numpy(outputs["uncertainty"]),
            "inference_ms": latency_ms,
        }

    def infer_batch(self, images_bgr: list[np.ndarray]) -> list[dict]:
        """Run inference on a list of BGR images. Returns list of result dicts."""
        return [self.infer(img) for img in images_bgr]
