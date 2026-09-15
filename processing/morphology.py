"""
processing/morphology.py
Morphological operations: thresholding, opening, closing, and dilation.
"""
import cv2
import numpy as np

def apply_morphology(image_bgr: np.ndarray) -> np.ndarray:
    """Apply morphological operations to isolate pothole candidate structures."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closing, kernel, iterations=1)
    
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

# Alias
morphological_processing = apply_morphology
