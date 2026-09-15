"""
scripts/live_camera.py
Real-Time Continuous Live Webcam Pothole Detection & Avoidance.

Run with:
    python scripts/live_camera.py
    python scripts/live_camera.py --camera 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root in sys.path
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import InferencePipeline, ACTION_NAMES, PipelineResult


def draw_hud_overlay(frame: np.ndarray, res: PipelineResult, fps: float) -> np.ndarray:
    """Draw automotive HUD overlays directly onto the video frame."""
    H, W = frame.shape[:2]
    canvas = frame.copy()

    # 1. Overlay Segmentation mask in translucent red
    seg = res.segmentation
    if seg.shape[:2] != (H, W):
        seg = cv2.resize(seg, (W, H))

    mask = (seg >= 0.5)
    canvas[mask] = (canvas[mask] * 0.45 + np.array([40, 40, 220]) * 0.55).astype(np.uint8)

    # 2. Draw contours & bounding boxes
    binary_mask = (seg >= 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(canvas, f"POTHOLE #{i+1}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # 3. Top Action Banner
    action = res.final_action
    action_name = ACTION_NAMES.get(action, "Maintain Lane").upper()
    is_override = (res.srl_decision is not None and not res.srl_decision.accepted)
    is_brake = (res.srl_decision and res.srl_decision.override_reason and "EMERGENCY" in res.srl_decision.override_reason)

    if is_brake:
        banner_color = (0, 0, 200) # Red
        banner_text = "EMERGENCY BRAKE - HAZARD AHEAD"
    elif action == 1:
        banner_color = (0, 200, 0) # Green
        banner_text = f"<-- SHIFT LEFT ({ 'SRL OVERRIDE' if is_override else 'RL POLICY' })"
    elif action == 2:
        banner_color = (0, 200, 0) # Green
        banner_text = f"SHIFT RIGHT --> ({ 'SRL OVERRIDE' if is_override else 'RL POLICY' })"
    else:
        banner_color = (220, 150, 0) # Cyan/Blue
        banner_text = "^^ MAINTAIN LANE (SAFE TRAJECTORY) ^^"

    # Banner header background
    cv2.rectangle(canvas, (0, 0), (W, 55), (15, 15, 20), -1)
    cv2.rectangle(canvas, (0, 0), (W, 55), banner_color, 2)
    cv2.putText(canvas, banner_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, banner_color, 2, cv2.LINE_AA)

    # 4. Telemetry Bottom Strip
    severity = float(res.state_vector[6]) if len(res.state_vector) > 6 else 0.0
    cv2.rectangle(canvas, (0, H - 35), (W, H), (15, 15, 20), -1)
    
    tel_str = f"FPS: {fps:.1f} | Latency: {res.inference_ms:.1f}ms | Potholes: {len(contours)} | Severity: {severity:.2f}"
    cv2.putText(canvas, tel_str, (15, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Real-Time Continuous Live Webcam Pothole Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" 🚗 POTHOLEGUARD-AI: REAL-TIME CONTINUOUS LIVE CAMERA")
    print("=" * 60)
    print(f" Connecting to Camera index: {args.camera}...")
    print(" Press 'q' or 'ESC' on the camera window to exit.")
    print("=" * 60 + "\n")

    pipeline = InferencePipeline(config_dir=str(ROOT / "configs"))
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {args.camera}. Please check connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = 0.0
    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame from camera.")
                break

            # Run inference pipeline on live frame
            res = pipeline.run(frame)

            # Calculate FPS
            curr_time = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
            prev_time = curr_time

            # Draw HUD
            hud_frame = draw_hud_overlay(frame, res, fps)

            # Show live continuous video window
            cv2.imshow("PotholeGuard-AI | Live Camera HUD", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nLive camera stopped.")


if __name__ == "__main__":
    main()
