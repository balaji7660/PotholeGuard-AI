"""
Shape and Morphology Analyzer for PotholeGuard-AI.
Extracts geometric irregularity metrics:
- Circularity = 4 * pi * Area / (Perimeter^2)
- Elongation / Aspect Ratio
- Boundary complexity / Convexity ratio
- Severity impact score
"""
import cv2
import numpy as np
import math
from typing import Dict, Any

class ShapeAnalyzer:
    def analyze_shape(self, contour: np.ndarray, area_px: int) -> Dict[str, Any]:
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter == 0 or area_px == 0:
            return {
                "circularity": 1.0,
                "elongation": 1.0,
                "boundary_complexity": 0.0,
                "shape_severity_factor": 0.5
            }

        # Circularity (1.0 for perfect circle, lower for irregular jagged potholes)
        circularity = float((4.0 * math.pi * area_px) / (perimeter ** 2))
        circularity = max(0.01, min(1.0, circularity))

        # Convex Hull and Convexity
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area_px / max(1.0, hull_area))

        # Fitted ellipse for elongation
        if len(contour) >= 5:
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
                elongation = float(ma / max(1.0, MA))
            except Exception:
                elongation = 1.0
        else:
            elongation = 1.0

        # Boundary complexity: high for jagged edges
        boundary_complexity = round(1.0 - solidity, 3)

        # Irregular, jagged, elongated potholes cause worse handlebar destabilization
        shape_severity = 0.4 * (1.0 - circularity) + 0.3 * min(1.0, elongation / 3.0) + 0.3 * boundary_complexity
        shape_severity = max(0.0, min(1.0, float(shape_severity)))

        return {
            "circularity": round(circularity, 3),
            "elongation": round(elongation, 2),
            "solidity": round(solidity, 3),
            "boundary_complexity": boundary_complexity,
            "shape_severity_factor": round(shape_severity, 3)
        }
