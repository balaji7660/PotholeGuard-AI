"""
gui/app.py
Pothole Detection System GUI (inspired by avxway/pothole-detection-system).
Features clean 8-button interface with step-wise visualization and live camera detection.
"""
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root in sys.path
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.enhancement import enhance_image
from processing.restoration import restore_image
from processing.morphology import morphological_processing
from processing.segmentation import segment_potholes
from app.pipeline import InferencePipeline, ACTION_NAMES
from scripts.live_camera import draw_hud_overlay


class PotholeDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pothole Detection")
        self.root.geometry("340x480")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")

        self.file_path = None
        self.file_type = None
        self.image = None

        # Load perception and safety pipeline
        self.pipeline = InferencePipeline(config_dir=str(ROOT / "configs"))

        self._create_widgets()

    def _create_widgets(self) -> None:
        frame = tk.Frame(self.root, bg="#f0f0f0", padx=30, pady=25)
        frame.pack(fill=tk.BOTH, expand=True)

        buttons = [
            ("Upload Image/Video", self.upload_file),
            ("Enhancement", self.show_enhancement),
            ("Restoration", self.show_restoration),
            ("Morphological Processing", self.show_morphology),
            ("Segmentation", self.show_segmentation),
            ("Detected Potholes (Image)", self.detect_image_action),
            ("View Detected Potholes (Video)", self.detect_video_action),
            ("Start Live Camera", self.detect_camera_action),
        ]

        for text, cmd in buttons:
            btn = tk.Button(
                frame,
                text=text,
                command=cmd,
                font=("Arial", 10),
                bg="#ffffff",
                fg="#000000",
                activebackground="#e0e0e0",
                relief=tk.RAISED,
                bd=1,
                padx=10,
                pady=6,
                cursor="hand2"
            )
            btn.pack(fill=tk.X, pady=6)

    def upload_file(self) -> None:
        filetypes = [
            ("Image & Video Files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
            ("Images", "*.jpg *.jpeg *.png *.bmp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        if not path:
            return

        self.file_path = path
        ext = Path(path).suffix.lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.file_type = "image"
            self.image = cv2.imread(path)
            messagebox.showinfo("Uploaded", f"Image selected: {Path(path).name}")
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            self.file_type = "video"
            self.image = None
            messagebox.showinfo("Uploaded", f"Video selected: {Path(path).name}\nClick 'View Detected Potholes (Video)' to run.")

    def _get_image(self) -> np.ndarray:
        if self.image is not None:
            return self.image
        # Fallback to default demo pothole image
        from app.demo_images import get_demo_image
        self.image = get_demo_image("center_pothole")
        return self.image

    def show_enhancement(self) -> None:
        img = self._get_image()
        enhanced = enhance_image(img)
        
        # Display with Matplotlib for clean, non-blocking window
        plt.figure("Image Enhancement (CLAHE)", figsize=(8, 4.5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Enhanced Image (CLAHE)")
        plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()

    def show_restoration(self) -> None:
        img = self._get_image()
        restored = restore_image(img)

        plt.figure("Image Restoration (Bilateral & Gaussian)", figsize=(8, 4.5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Restored / Denoised Image")
        plt.imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def show_morphology(self) -> None:
        img = self._get_image()
        morph = morphological_processing(img)

        plt.figure("Morphological Processing", figsize=(8, 4.5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Morphological Mask (Open/Close/Dilate)")
        plt.imshow(cv2.cvtColor(morph, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def show_segmentation(self) -> None:
        img = self._get_image()
        seg = segment_potholes(img)

        plt.figure("Pothole Segmentation", figsize=(8, 4.5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Segmented Potholes with Contours")
        plt.imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def detect_image_action(self) -> None:
        img = self._get_image()
        res = self.pipeline.run(img)
        hud_img = draw_hud_overlay(img, res, fps=30.0)

        cv2.imshow("Detected Potholes (Image) | Press any key to close", hud_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Detected Potholes (Image) | Press any key to close")

    def detect_video_action(self) -> None:
        if not self.file_path or self.file_type != "video":
            messagebox.showwarning("No Video", "Please click 'Upload Image/Video' and select a video file (.mp4, .avi) first.")
            return

        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            res = self.pipeline.run(frame)
            hud_frame = draw_hud_overlay(frame, res, fps)

            cv2.imshow("View Detected Potholes (Video) | Press 'q' to Exit", hud_frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_camera_action(self) -> None:
        # Open webcam with Windows DirectShow backend
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera 0. Please verify your webcam is connected.")
            return

        fps = 0.0
        prev_time = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Calculate FPS
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1.0 / max(1e-4, time_diff)
            prev_time = curr_time

            res = self.pipeline.run(frame)
            hud_frame = draw_hud_overlay(frame, res, fps)

            cv2.imshow("Live Camera Pothole Detection | Press 'q' to Stop", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()
