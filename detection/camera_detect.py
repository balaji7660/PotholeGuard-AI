"""
detection/camera_detect.py
Continuous live camera pothole detection.
"""
import cv2
from scripts.live_camera import draw_hud_overlay

def detect_camera(model):
    """Open continuous webcam stream with real-time detection at 30 FPS."""
    # Open camera with Windows DirectShow backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam device index 0.")
        return

    fps = 0.0
    prev_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Measure FPS
        curr_time = cv2.getTickCount()
        time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
        fps = 1.0 / max(1e-4, time_diff)
        prev_time = curr_time

        # Run pipeline
        res = model.run(frame)
        hud_frame = draw_hud_overlay(frame, res, fps)

        cv2.imshow("Live Camera Pothole Detection | Press 'q' to Stop", hud_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
