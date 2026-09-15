"""
processing/restoration.py
Image restoration using Bilateral and Gaussian noise filtering.
"""
import cv2
import numpy as np

def restore(image_bgr: np.ndarray) -> np.ndarray:
    """Restore image by removing road grain noise while preserving pothole edges."""
    bilateral = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.GaussianBlur(bilateral, (5, 5), 0)

# Alias
restore_image = restore
