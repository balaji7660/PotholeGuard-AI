"""
detection/image_detect.py
Pothole detection on static images.
"""
import cv2
import matplotlib.pyplot as plt
from scripts.live_camera import draw_hud_overlay

def detect_image(model, image):
    """Run detection on image and display with HUD bounding boxes."""
    if image is None:
        return

    res = model.run(image)
    detected_frame = draw_hud_overlay(image, res, fps=30.0)

    cv2.imshow("Detected Potholes (Image) | Press any key to close", detected_frame)
    cv2.waitKey(0)
    cv2.destroyWindow("Detected Potholes (Image) | Press any key to close")
