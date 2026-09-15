"""
Size Estimator module for PotholeGuard-AI.
Extracts geometric dimensions (width, height, area, aspect ratio) from pothole binary masks/contours.
Supports:
- Relative Size Classification (SMALL, MEDIUM, LARGE, VERY LARGE)
- Optional Calibrated Estimation in physical units (cm / meters) when camera parameters are provided.
"""
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple

class SizeEstimator:
    def __init__(self, calibration_config: Optional[Dict[str, Any]] = None):
        self.calibration = calibration_config or {}
        self.is_calibrated = bool(self.calibration.get("calibrated", False))
        self.pixel_to_cm_ratio = float(self.calibration.get("pixel_to_cm_ratio", 0.0))

    def update_calibration(self, calibration_config: Dict[str, Any]):
        self.calibration = calibration_config
        self.is_calibrated = bool(calibration_config.get("calibrated", False))
        self.pixel_to_cm_ratio = float(calibration_config.get("pixel_to_cm_ratio", 0.0))

    def estimate_size(self, mask: np.ndarray, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Estimate size parameters from mask and bounding box.
        Args:
            mask: Binary mask of single pothole instance (uint8 0 or 255)
            bbox: (x1, y1, x2, y2)
            frame_shape: (H, W)
        """
        x1, y1, x2, y2 = bbox
        width_px = max(1, x2 - x1)
        height_px = max(1, y2 - y1)
        area_px = int(np.sum(mask > 0)) if mask is not None else width_px * height_px
        aspect_ratio = round(width_px / float(height_px), 2)
        
        frame_area = frame_shape[0] * frame_shape[1]
        area_ratio = area_px / float(frame_area)
        
        # Classification thresholds based on image frame proportion
        if area_ratio < 0.015:
            size_class = "SMALL"
            size_score = 0.25
        elif area_ratio < 0.05:
            size_class = "MEDIUM"
            size_score = 0.55
        elif area_ratio < 0.12:
            size_class = "LARGE"
            size_score = 0.85
        else:
            size_class = "VERY LARGE"
            size_score = 1.0

        result = {
            "width_px": int(width_px),
            "height_px": int(height_px),
            "area_px": int(area_px),
            "aspect_ratio": float(aspect_ratio),
            "size_class": size_class,
            "size_score": float(size_score),
            "is_calibrated": self.is_calibrated
        }

        if self.is_calibrated and self.pixel_to_cm_ratio > 0:
            result["width_cm"] = round(width_px * self.pixel_to_cm_ratio, 1)
            result["height_cm"] = round(height_px * self.pixel_to_cm_ratio, 1)
            result["area_cm2"] = round(area_px * (self.pixel_to_cm_ratio ** 2), 1)
            result["size_label"] = f"Calibrated: {result['width_cm']}x{result['height_cm']} cm"
        else:
            result["size_label"] = f"Relative size: {size_class}"

        return result
