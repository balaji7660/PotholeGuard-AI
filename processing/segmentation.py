"""
processing/segmentation.py
Pothole region segmentation and contour boundary detection.
"""
import cv2
import numpy as np


def segment_potholes(image_bgr: np.ndarray) -> np.ndarray:
    """Segment pothole regions and overlay segmentation mask with red boundary contours."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Adaptive / Otsu Thresholding
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Restrict to road surface (bottom 2/3 of frame)
    H, W = binary.shape[:2]
    binary[:H//3, :] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Overlay on original image
    vis = image_bgr.copy()
    mask = (cleaned > 0)
    vis[mask] = (vis[mask] * 0.4 + np.array([40, 40, 220]) * 0.6).astype(np.uint8)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    
    return vis
