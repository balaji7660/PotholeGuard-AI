"""
gui/app.py
Tkinter GUI matching avxway/pothole-detection-system.
"""
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import cv2

from detection.model import load_model
from utils.file_handler import upload_file

from processing.enhancement import enhance
from processing.restoration import restore
from processing.morphology import apply_morphology
from processing.segmentation import segment

from detection.image_detect import detect_image
from detection.video_detect import detect_video
from detection.camera_detect import detect_camera


class PotholeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pothole Detection")
        self.root.geometry("320x460")
        self.root.resizable(False, False)

        self.model = load_model()
        self.file_path = None
        self.file_type = None
        self.image = None

        self._create_widgets()

    def _create_widgets(self):
        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        buttons = [
            ("Upload Image/Video", self.upload),
            ("Enhancement", self.enhance_image),
            ("Restoration", self.restore_image),
            ("Morphological Processing", self.morphology_image),
            ("Segmentation", self.segment_image),
            ("Detected Potholes (Image)", self.detect_img),
            ("View Detected Potholes (Video)", self.detect_vid),
            ("Start Live Camera", self.detect_cam),
        ]

        for text, cmd in buttons:
            btn = tk.Button(frame, text=text, command=cmd, width=26, pady=5)
            btn.pack(pady=4)

    def upload(self):
        self.file_path, self.file_type, self.image = upload_file()
        if self.file_path:
            messagebox.showinfo("Success", f"Loaded: {self.file_path.split('/')[-1]}")

    def _get_image(self):
        if self.image is not None:
            return self.image
        from app.demo_images import get_demo_image
        self.image = get_demo_image("center_pothole")
        return self.image

    def enhance_image(self):
        img = self._get_image()
        enhanced = enhance(img)

        plt.figure("Enhancement (CLAHE)", figsize=(8, 4.5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Enhanced Image")
        plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def restore_image(self):
        img = self._get_image()
        restored = restore(img)

        plt.figure("Restoration", figsize=(8, 4.5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Restored Image")
        plt.imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def morphology_image(self):
        img = self._get_image()
        morph = apply_morphology(img)

        plt.figure("Morphological Processing", figsize=(8, 4.5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Morphological Mask")
        plt.imshow(cv2.cvtColor(morph, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def segment_image(self):
        img = self._get_image()
        seg = segment(img)

        plt.figure("Segmentation", figsize=(8, 4.5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Segmented Image")
        plt.imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def detect_img(self):
        img = self._get_image()
        detect_image(self.model, img)

    def detect_vid(self):
        if not self.file_path or self.file_type != "video":
            messagebox.showwarning("No Video", "Please click 'Upload Image/Video' and select a video file first.")
            return
        detect_video(self.model, self.file_path)

    def detect_cam(self):
        detect_camera(self.model)
