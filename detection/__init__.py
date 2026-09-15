from detection.model import load_model
from detection.image_detect import detect_image
from detection.video_detect import detect_video
from detection.camera_detect import detect_camera

__all__ = ["load_model", "detect_image", "detect_video", "detect_camera"]
