"""
Image Quality Assessment Module for PotholeGuard-AI.
Evaluates:
- Sharpness / Blur (Laplacian variance)
- Brightness / Luminance distribution
- Contrast (RMS contrast)
- Glare / Over-saturation detection
- Motion blur estimation

Outputs Image Quality tier: HIGH, MEDIUM, LOW and a quality confidence weight.
"""
import cv2
import numpy as np
from typing import Dict, Any

class ImageQualityAssessor:
    def __init__(self, blur_threshold: float = 60.0, low_light_threshold: float = 40.0, glare_threshold: float = 230.0):
        self.blur_threshold = blur_threshold
        self.low_light_threshold = low_light_threshold
        self.glare_threshold = glare_threshold

    def evaluate_quality(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur score via Laplacian variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = lap_var < self.blur_threshold

        # 2. Brightness (Mean luminance)
        mean_brightness = float(np.mean(gray))
        is_dark = mean_brightness < self.low_light_threshold
        is_glare = float(np.mean(gray > self.glare_threshold)) > 0.15

        # 3. RMS Contrast
        contrast = float(np.std(gray))
        is_low_contrast = contrast < 25.0

        # Score synthesis in [0, 1]
        quality_score = 1.0
        degradation_reasons = []

        if is_blurry:
            quality_score -= 0.35
            degradation_reasons.append("MOTION_BLUR_OR_VIBRATION")
        if is_dark:
            quality_score -= 0.30
            degradation_reasons.append("LOW_LIGHT_ENVIRONMENT")
        if is_glare:
            quality_score -= 0.25
            degradation_reasons.append("SOLAR_GLARE_OR_OVEREXPOSURE")
        if is_low_contrast:
            quality_score -= 0.15
            degradation_reasons.append("POOR_CONTRAST")

        quality_score = max(0.1, min(1.0, quality_score))

        if quality_score >= 0.75:
            quality_tier = "HIGH"
        elif quality_score >= 0.45:
            quality_tier = "MEDIUM"
        else:
            quality_tier = "LOW"

        return {
            "quality_tier": quality_tier,
            "quality_score": round(quality_score, 2),
            "laplacian_variance": round(lap_var, 1),
            "mean_brightness": round(mean_brightness, 1),
            "rms_contrast": round(contrast, 1),
            "degradation_reasons": degradation_reasons,
            "is_reliable": quality_tier in ["HIGH", "MEDIUM"]
        }
