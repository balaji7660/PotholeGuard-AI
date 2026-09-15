"""
Uncertainty Processor module for PotholeGuard-AI.
Calculates pothole-level uncertainty score from epistemic/aleatoric maps.
Prevents false 'SAFE' classifications and triggers 'UNCERTAIN - USE CAUTION' alerts
when epistemic variance exceeds critical reliability thresholds.
"""
import numpy as np
from typing import Dict, Any

class UncertaintyProcessor:
    def __init__(self, high_uncertainty_threshold: float = 0.45):
        self.high_uncertainty_threshold = high_uncertainty_threshold

    def process_uncertainty(self, uncertainty_map: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract uncertainty metrics inside pothole region.
        """
        pothole_pixels = uncertainty_map[mask > 0]
        if len(pothole_pixels) == 0:
            mean_unc = float(np.mean(uncertainty_map))
        else:
            mean_unc = float(np.mean(pothole_pixels))

        is_high = mean_unc >= self.high_uncertainty_threshold
        # Confidence is inverse of uncertainty
        confidence = max(0.0, min(1.0, 1.0 - mean_unc))

        return {
            "uncertainty": round(mean_unc, 3),
            "confidence": round(confidence, 3),
            "is_high_uncertainty": is_high,
            "uncertainty_label": "HIGH UNCERTAINTY" if is_high else "CONFIDENT"
        }
