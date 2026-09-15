"""
utils/file_handler.py
File dialog handler for uploading road images and videos.
"""
from tkinter import filedialog
import cv2

def upload_file():
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Supported Media", "*.jpg;*.jpeg;*.png;*.bmp;*.mp4;*.avi;*.mov"),
            ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"),
            ("Video Files", "*.mp4;*.avi;*.mov"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return None, None, None

    if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        image = cv2.imread(file_path)
        return file_path, "image", image
    elif file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return file_path, "video", None
    
    return file_path, "unknown", None
