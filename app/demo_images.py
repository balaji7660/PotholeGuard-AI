"""
app/demo_images.py
Generates synthetic demo pothole images for use when no real dataset is available.
These are clearly labeled as synthetic demo images — NOT real benchmark data.
"""
from __future__ import annotations

from typing import List
import cv2
import numpy as np

# Demo image registry
DEMO_IMAGE_NAMES = [
    "demo_pothole_centre",
    "demo_pothole_left",
    "demo_pothole_right",
    "demo_multi_pothole",
    "demo_no_pothole",
    "demo_severe_pothole",
    "demo_wet_road",
]


def _make_road_background(h: int = 512, w: int = 512) -> np.ndarray:
    """Asphalt-grey road background with lane markings."""
    img = np.full((h, w, 3), (70, 72, 68), dtype=np.uint8)

    # Add texture (random noise)
    noise = np.random.randint(-12, 12, (h, w, 3), dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 30, 120).astype(np.uint8)

    # Gradient (brighter at horizon)
    gradient = np.linspace(1.0, 0.7, h, dtype=np.float32)[:, np.newaxis, np.newaxis]
    img = (img * gradient).astype(np.uint8)

    # Lane markings (white)
    lx0 = int(w * 0.25)
    lx1 = int(w * 0.75)
    cv2.line(img, (lx0, 0), (lx0, h), (200, 200, 200), 4)
    cv2.line(img, (lx1, 0), (lx1, h), (200, 200, 200), 4)

    # Dashed centre line
    for y in range(0, h, 40):
        cv2.line(img, (w//2, y), (w//2, min(y+20, h)), (220, 220, 80), 2)

    return img


def _draw_pothole(img: np.ndarray, cx: int, cy: int, rx: int, ry: int,
                  depth: float = 0.5) -> np.ndarray:
    """Draw a realistic-looking pothole ellipse."""
    img = img.copy()
    # Dark fill
    darkness = int(20 + (1 - depth) * 30)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (darkness, darkness, darkness), -1)
    # Edge shadow
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (40, 38, 35), 3)
    # Highlight rim
    cv2.ellipse(img, (cx-3, cy-3), (rx-3, ry-3), 10, 200, 280,
                (90, 88, 84), 2)
    return img


def get_demo_image(name: str) -> np.ndarray:
    """
    Generate a synthetic demo BGR image for the given name.
    Images are procedurally generated — NOT from any real dataset.
    """
    np.random.seed(hash(name) % 2**31)
    h, w = 512, 512
    img  = _make_road_background(h, w)

    cx_mid = w // 2
    cy_mid = int(h * 0.65)

    if name == "demo_pothole_centre":
        img = _draw_pothole(img, cx_mid, cy_mid, 60, 40, depth=0.3)

    elif name == "demo_pothole_left":
        img = _draw_pothole(img, int(w * 0.35), cy_mid, 50, 35, depth=0.4)

    elif name == "demo_pothole_right":
        img = _draw_pothole(img, int(w * 0.65), cy_mid, 55, 38, depth=0.35)

    elif name == "demo_multi_pothole":
        img = _draw_pothole(img, int(w * 0.38), int(h * 0.60), 40, 28, depth=0.4)
        img = _draw_pothole(img, int(w * 0.62), int(h * 0.70), 45, 30, depth=0.3)
        img = _draw_pothole(img, cx_mid,          int(h * 0.50), 30, 20, depth=0.5)

    elif name == "demo_no_pothole":
        pass  # Clean road

    elif name == "demo_severe_pothole":
        img = _draw_pothole(img, cx_mid, cy_mid, 90, 65, depth=0.1)
        # Water fill effect
        cv2.ellipse(img, (cx_mid, cy_mid), (85, 60), 0, 0, 360, (40, 35, 20), -1)

    elif name == "demo_wet_road":
        img = _draw_pothole(img, int(w * 0.45), int(h * 0.68), 55, 38, depth=0.35)
        # Wet sheen
        wet = np.ones_like(img, dtype=np.float32) * np.array([1.05, 1.02, 0.98])
        img = np.clip((img * wet).astype(np.uint8), 0, 255)

    # Add sky/horizon area (top 1/3)
    sky = np.full((h//3, w, 3), (140, 155, 160), dtype=np.uint8)
    sky_noise = np.random.randint(-5, 5, sky.shape, dtype=np.int16)
    sky = np.clip(sky.astype(np.int16) + sky_noise, 100, 200).astype(np.uint8)
    img[:h//3] = sky

    # Label as DEMO
    cv2.putText(img, "DEMO IMAGE - Not real benchmark data",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 80, 80), 1)

    return img   # BGR


def list_demo_images() -> List[str]:
    return DEMO_IMAGE_NAMES
