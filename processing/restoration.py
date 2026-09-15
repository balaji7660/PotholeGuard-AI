"""
processing/restoration.py
Image restoration stage using Bilateral and Gaussian noise filtering.
"""
import cv2
import numpy as np


def restore_image(image_bgr: np.ndarray) -> np.ndarray:
    """Restore image by removing road grain noise while preserving pothole edges."""
    # Bilateral filter for edge-preserving smoothing
    bilateral = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    # Gaussian blur for high-frequency noise removal
    restored = cv2.GaussianBlur(bilateral, (5, 5), 0)
    return restored
