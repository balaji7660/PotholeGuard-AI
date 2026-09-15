"""
processing/morphology.py
Morphological operations: thresholding, opening, closing, and dilation.
"""
import cv2
import numpy as np


def morphological_processing(image_bgr: np.ndarray) -> np.ndarray:
    """Apply morphological operations to isolate pothole candidate structures."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological kernels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Morphological Opening (removes small road speckles)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Morphological Closing (fills interior pothole gaps)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilation for contour boundary clarity
    dilated = cv2.dilate(closing, kernel, iterations=1)
    
    # Return 3-channel visualization
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
