"""
data/augmentation.py
Photometric and geometric augmentation pipeline using albumentations.
Supports CutMix for segmentation tasks.
"""
from __future__ import annotations

import random
from typing import Tuple

import albumentations as A
import numpy as np
from omegaconf import DictConfig


class PotholeAugmentor:
    """
    Applies photometric and geometric augmentations to image/mask/depth triplets.

    Parameters
    ----------
    cfg : DictConfig
        Full dataset configuration (dataset_config.yaml).
    train : bool
        If False, no augmentation is applied (validation/test).
    """

    def __init__(self, cfg: DictConfig, train: bool = True) -> None:
        self.train = train
        self.aug_cfg = cfg.augmentation
        self.cutmix_cfg = cfg.augmentation.cutmix
        self._transform = self._build_transform()

    # ------------------------------------------------------------------
    # Transform builder
    # ------------------------------------------------------------------

    def _build_transform(self) -> A.Compose:
        """Construct the albumentations pipeline."""
        if not self.train or not self.aug_cfg.enabled:
            # Val/test: only resize normalisation (done outside augmentor)
            return A.Compose([A.NoOp()])

        transforms: list[A.BasicTransform] = []

        if self.aug_cfg.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))

        if self.aug_cfg.vertical_flip:
            transforms.append(A.VerticalFlip(p=0.3))

        if self.aug_cfg.random_rotate.enabled:
            transforms.append(
                A.Rotate(limit=self.aug_cfg.random_rotate.limit, p=0.5, border_mode=0)
            )

        if self.aug_cfg.color_jitter.enabled:
            jitter = self.aug_cfg.color_jitter
            transforms.append(
                A.ColorJitter(
                    brightness=jitter.brightness,
                    contrast=jitter.contrast,
                    saturation=jitter.saturation,
                    hue=jitter.hue,
                    p=0.5,
                )
            )

        if self.aug_cfg.gaussian_blur.enabled:
            transforms.append(
                A.GaussianBlur(
                    blur_limit=(3, self.aug_cfg.gaussian_blur.blur_limit),
                    p=0.3,
                )
            )

        # Additional augmentations for robustness
        transforms += [
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(p=0.2),
            A.RandomGamma(p=0.2),
        ]

        return A.Compose(
            transforms,
            additional_targets={
                "depth": "image",   # treat depth like an image for geometric transforms
            },
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentation.

        Parameters
        ----------
        image : np.ndarray  (H, W, 3) uint8 RGB
        mask  : np.ndarray  (H, W)    float32 {0,1}
        depth : np.ndarray  (H, W)    float32

        Returns
        -------
        Augmented (image, mask, depth) with same dtypes.
        """
        if not self.train or not self.aug_cfg.enabled:
            return image, mask, depth

        # albumentations expects mask as uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        result = self._transform(image=image, mask=mask_uint8, depth=depth)
        aug_image = result["image"]
        aug_mask  = (result["mask"] > 127).astype(np.float32)
        aug_depth = result["depth"]

        # CutMix (applied separately, after geometric augmentation)
        if self.cutmix_cfg.enabled and random.random() < self.cutmix_cfg.prob:
            aug_image, aug_mask, aug_depth = self._cutmix_self(
                aug_image, aug_mask, aug_depth
            )

        return aug_image, aug_mask, aug_depth

    # ------------------------------------------------------------------
    # CutMix implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_box(h: int, w: int, lam: float) -> Tuple[int, int, int, int]:
        """Sample a random bounding box for CutMix."""
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)
        return x1, y1, x2, y2

    def _cutmix_self(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Self-CutMix: paste a flipped version of a region onto the image.
        Used when we don't have a second sample (single-item call).
        """
        h, w = image.shape[:2]
        alpha = float(self.cutmix_cfg.alpha)
        lam = np.random.beta(alpha, alpha)
        x1, y1, x2, y2 = self._sample_box(h, w, lam)

        # Use horizontally-flipped self as the "other" sample
        other_image = image[:, ::-1, :].copy()
        other_mask  = mask[:, ::-1].copy()
        other_depth = depth[:, ::-1].copy()

        image = image.copy()
        mask  = mask.copy()
        depth = depth.copy()

        image[y1:y2, x1:x2] = other_image[y1:y2, x1:x2]
        mask[y1:y2, x1:x2]  = other_mask[y1:y2, x1:x2]
        depth[y1:y2, x1:x2] = other_depth[y1:y2, x1:x2]

        return image, mask, depth
