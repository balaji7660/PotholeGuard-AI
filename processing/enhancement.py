"""
processing/enhancement.py
Image enhancement stage using CLAHE and contrast stretching.
"""
import cv2
import numpy as np


def enhance_image(image_bgr: np.ndarray) -> np.ndarray:
    """Enhance road contrast and details using CLAHE on LAB color space."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr
