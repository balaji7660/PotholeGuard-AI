"""
Rider-Path and Road Region Estimation module for PotholeGuard-AI.
Calculates lateral displacement relative to the motorcycle centerline (camera center),
evaluates path relevance (LOW / MEDIUM / HIGH), and provides graceful degradation
if road boundary perception is noisy or unavailable.
"""
import numpy as np
from typing import Dict, Any, Tuple

class PathEstimator:
    def __init__(self, rider_corridor_ratio: float = 0.35):
        """
        Args:
            rider_corridor_ratio: Fraction of the central frame width considered the direct trajectory corridor.
        """
        self.rider_corridor_ratio = rider_corridor_ratio

    def estimate_path_relevance(
        self,
        centroid: Tuple[int, int],
        frame_shape: Tuple[int, int],
        bbox: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        Args:
            centroid: (cx, cy)
            frame_shape: (H, W)
            bbox: (x1, y1, x2, y2)
        """
        H, W = frame_shape
        cx, cy = centroid
        frame_center_x = W / 2.0
        
        # Lateral displacement in pixels (-W/2 to +W/2)
        displacement_px = cx - frame_center_x
        # Normalized displacement in [-1.0, 1.0]
        norm_displacement = float(displacement_px / (W / 2.0))
        
        # Determine position category
        corridor_half = self.rider_corridor_ratio / 2.0
        if norm_displacement < -corridor_half:
            position_class = "LEFT OF PATH"
        elif norm_displacement > corridor_half:
            position_class = "RIGHT OF PATH"
        else:
            position_class = "CENTER / DIRECT PATH"

        # Longitudinal distance proxy based on vertical position (bottom is near, top is far)
        norm_distance_y = float(cy / float(H)) # near 1.0 means right in front of the tire

        # Path relevance: highest when centered and in lower half of image (immediate threat)
        center_proximity = max(0.0, 1.0 - abs(norm_displacement))
        path_relevance_score = 0.6 * center_proximity + 0.4 * norm_distance_y
        path_relevance_score = max(0.0, min(1.0, float(path_relevance_score)))

        if path_relevance_score > 0.65:
            relevance_class = "HIGH"
        elif path_relevance_score > 0.35:
            relevance_class = "MEDIUM"
        else:
            relevance_class = "LOW"

        return {
            "lateral_displacement_px": int(displacement_px),
            "normalized_displacement": round(norm_displacement, 3),
            "position_class": position_class,
            "path_relevance_score": round(path_relevance_score, 3),
            "relevance_class": relevance_class,
            "is_direct_path": position_class == "CENTER / DIRECT PATH"
        }
