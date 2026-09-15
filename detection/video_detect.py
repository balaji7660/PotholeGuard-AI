"""
detection/video_detect.py
Pothole detection on video files.
"""
import cv2
from scripts.live_camera import draw_hud_overlay

def detect_video(model, video_path):
    """Run real-time detection on video file and play in OpenCV window."""
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        res = model.run(frame)
        hud_frame = draw_hud_overlay(frame, res, fps)

        cv2.imshow("Detected Potholes (Video) | Press 'q' to Exit", hud_frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
