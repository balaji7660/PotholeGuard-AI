"""
Depth Processor module for PotholeGuard-AI.
Analyzes relative depth inside pothole segmentation masks.
Calculates:
- Mean relative depth
- Minimum relative depth
- Depth variance
- Classification into SHALLOW / MEDIUM / DEEP
- Optional calibrated distance estimation in meters if camera height & tilt are known.
"""
import numpy as np
from typing import Dict, Any, Optional

class DepthProcessor:
    def __init__(self, calibration_config: Optional[Dict[str, Any]] = None):
        self.calibration = calibration_config or {}
        self.is_calibrated = bool(self.calibration.get("calibrated", False))
        self.depth_scale = float(self.calibration.get("depth_scale_meters", 1.0))

    def update_calibration(self, calibration_config: Dict[str, Any]):
        self.calibration = calibration_config
        self.is_calibrated = bool(calibration_config.get("calibrated", False))
        self.depth_scale = float(calibration_config.get("depth_scale_meters", 1.0))

    def process_depth(self, depth_map: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract depth statistics inside the pothole mask.
        Args:
            depth_map: 2D numpy array [0, 1]
            mask: 2D binary numpy array (0 or 255)
        """
        pothole_pixels = depth_map[mask > 0]
        if len(pothole_pixels) == 0:
            return {
                "mean_depth": 0.0,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "depth_variance": 0.0,
                "depth_class": "SHALLOW",
                "depth_score": 0.0,
                "depth_label": "Relative depth: SHALLOW",
                "is_calibrated": self.is_calibrated
            }

        mean_depth = float(np.mean(pothole_pixels))
        min_depth = float(np.min(pothole_pixels))
        max_depth = float(np.max(pothole_pixels))
        depth_var = float(np.var(pothole_pixels))

        # Higher values indicate deeper depressions or closer critical proximity
        if mean_depth < 0.35:
            depth_class = "SHALLOW"
            depth_score = 0.3
        elif mean_depth < 0.70:
            depth_class = "MEDIUM"
            depth_score = 0.65
        else:
            depth_class = "DEEP"
            depth_score = 0.95

        result = {
            "mean_depth": round(mean_depth, 3),
            "min_depth": round(min_depth, 3),
            "max_depth": round(max_depth, 3),
            "depth_variance": round(depth_var, 4),
            "depth_class": depth_class,
            "depth_score": float(depth_score),
            "is_calibrated": self.is_calibrated
        }

        if self.is_calibrated:
            est_depth_m = round(mean_depth * self.depth_scale, 2)
            result["estimated_depth_m"] = est_depth_m
            result["depth_label"] = f"Estimated depth: {est_depth_m} m"
        else:
            result["depth_label"] = f"Relative depth: {depth_class}"

        return result
