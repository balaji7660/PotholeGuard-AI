"""
gui/app.py
Tkinter GUI for Pothole Detection & Avoidance System.
Implements step-wise DIP visualization, image/video detection, and continuous live camera.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Make root importable
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.enhancement import enhance_image
from processing.restoration import restore_image
from processing.morphology import morphological_processing
from processing.segmentation import segment_potholes
from app.pipeline import InferencePipeline, ACTION_NAMES, PipelineResult
from scripts.live_camera import draw_hud_overlay


class PotholeDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pothole Detection")
        self.root.geometry("420x540")
        self.root.resizable(False, False)
        self.root.configure(bg="#f8fafc")

        # Initialize pipeline instance
        self.pipeline = InferencePipeline(config_dir=str(ROOT / "configs"))

        # State variables
        self.current_image_path: Optional[str] = None
        self.current_video_path: Optional[str] = None
        self.current_image_bgr: Optional[np.ndarray] = None
        self.is_camera_running = False

        self._build_ui()

    def _build_ui(self) -> None:
        # Title Label
        title_frame = tk.Frame(self.root, bg="#f8fafc", pady=16)
        title_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            title_frame,
            text="Pothole Detection & Avoidance",
            font=("Helvetica", 14, "bold"),
            bg="#f8fafc",
            fg="#0f172a"
        )
        title_lbl.pack()

        sub_lbl = tk.Label(
            title_frame,
            text="Digital Image Processing & AI Perception",
            font=("Helvetica", 9),
            bg="#f8fafc",
            fg="#64748b"
        )
        sub_lbl.pack()

        # Button container
        btn_frame = tk.Frame(self.root, bg="#f8fafc", padx=30, pady=10)
        btn_frame.pack(fill=tk.BOTH, expand=True)

        buttons = [
            ("Upload Image/Video", self.upload_file, "#ffffff", "#0f172a"),
            ("Enhancement", self.show_enhancement, "#ffffff", "#0f172a"),
            ("Restoration", self.show_restoration, "#ffffff", "#0f172a"),
            ("Morphological Processing", self.show_morphology, "#ffffff", "#0f172a"),
            ("Segmentation", self.show_segmentation, "#ffffff", "#0f172a"),
            ("Detected Potholes (Image)", self.show_detected_image, "#ffffff", "#0f172a"),
            ("View Detected Potholes (Video)", self.show_detected_video, "#ffffff", "#0f172a"),
            ("Start Live Camera", self.start_live_camera, "#38bdf8", "#031320"),
        ]

        for text, cmd, bg_col, fg_col in buttons:
            btn = tk.Button(
                btn_frame,
                text=text,
                command=cmd,
                font=("Helvetica", 10, "bold" if "Live Camera" in text else "normal"),
                bg=bg_col,
                fg=fg_col,
                activebackground="#e2e8f0",
                relief=tk.RAISED,
                bd=1,
                cursor="hand2",
                height=1,
                pady=6
            )
            btn.pack(fill=tk.X, pady=5)

        # Status footer
        self.status_var = tk.StringVar(value="Status: Ready. Please upload an image/video or start camera.")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 8),
            bg="#e2e8f0",
            fg="#334155",
            anchor="w",
            padx=10,
            pady=4
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ── Button Handlers ────────────────────────────────────────────────────────

    def upload_file(self) -> None:
        """Open file dialog for image or video."""
        filetypes = [
            ("All Supported Media", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
            ("Images", "*.jpg *.jpeg *.png *.bmp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ]
        filepath = filedialog.askopenfilename(title="Select Road Image or Video", filetypes=filetypes)
        if not filepath:
            return

        ext = Path(filepath).suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.current_image_path = filepath
            self.current_video_path = None
            self.current_image_bgr = cv2.imread(filepath)
            self.status_var.set(f"Loaded Image: {Path(filepath).name}")
            messagebox.showinfo("Success", f"Image loaded successfully: {Path(filepath).name}\nClick Enhancement, Segmentation, or Detected Potholes.")
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            self.current_video_path = filepath
            self.current_image_path = None
            self.status_var.set(f"Loaded Video: {Path(filepath).name}")
            messagebox.showinfo("Success", f"Video loaded successfully: {Path(filepath).name}\nClick 'View Detected Potholes (Video)' to run detection.")

    def _ensure_image_loaded(self) -> bool:
        if self.current_image_bgr is None:
            # Fallback: offer default demo image if none loaded
            from app.demo_images import get_demo_image
            self.current_image_bgr = get_demo_image("center_pothole")
            self.status_var.set("Using demo road image (none uploaded).")
            return True
        return True

    def show_enhancement(self) -> None:
        """Run and display contrast enhancement."""
        if not self._ensure_image_loaded(): return
        self.status_var.set("Processing: Image Enhancement (CLAHE)...")
        enhanced = enhance_image(self.current_image_bgr)
        self._display_side_by_side("Original Image", self.current_image_bgr, "Enhanced Image (CLAHE)", enhanced)
        self.status_var.set("Status: Enhancement complete.")

    def show_restoration(self) -> None:
        """Run and display noise restoration."""
        if not self._ensure_image_loaded(): return
        self.status_var.set("Processing: Image Restoration (Bilateral + Gaussian)...")
        restored = restore_image(self.current_image_bgr)
        self._display_side_by_side("Original Image", self.current_image_bgr, "Restored / Denoised Image", restored)
        self.status_var.set("Status: Restoration complete.")

    def show_morphology(self) -> None:
        """Run and display morphological operations."""
        if not self._ensure_image_loaded(): return
        self.status_var.set("Processing: Morphological Operations (Opening/Closing)...")
        morph = morphological_processing(self.current_image_bgr)
        self._display_side_by_side("Original Image", self.current_image_bgr, "Morphological Processing", morph)
        self.status_var.set("Status: Morphology complete.")

    def show_segmentation(self) -> None:
        """Run and display pothole segmentation."""
        if not self._ensure_image_loaded(): return
        self.status_var.set("Processing: Pothole Segmentation...")
        seg = segment_potholes(self.current_image_bgr)
        self._display_side_by_side("Original Image", self.current_image_bgr, "Segmented Potholes", seg)
        self.status_var.set("Status: Segmentation complete.")

    def show_detected_image(self) -> None:
        """Run complete AI inference & draw bounding boxes + avoidance action."""
        if not self._ensure_image_loaded(): return
        self.status_var.set("Running Full Perception & Avoidance Pipeline...")
        
        t0 = time.perf_counter()
        res = self.pipeline.run(self.current_image_bgr)
        fps = 1.0 / max(1e-4, time.perf_counter() - t0)
        
        detected_vis = draw_hud_overlay(self.current_image_bgr, res, fps)
        
        cv2.imshow("Detected Potholes (Image) | PotholeGuard-AI", detected_vis)
        self.status_var.set(f"Detection Done. Action: {ACTION_NAMES.get(res.final_action)} | Potholes: {res.n_potholes}")
        cv2.waitKey(0)
        cv2.destroyWindow("Detected Potholes (Image) | PotholeGuard-AI")

    def show_detected_video(self) -> None:
        """Process video frame by frame with pothole detection and steering HUD."""
        if not self.current_video_path:
            messagebox.showwarning("No Video", "Please click 'Upload Image/Video' and select a video file first.")
            return

        self.status_var.set(f"Playing detected video: {Path(self.current_video_path).name} (Press 'q' to stop)")
        cap = cv2.VideoCapture(self.current_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            res = self.pipeline.run(frame)
            hud_frame = draw_hud_overlay(frame, res, fps)

            cv2.imshow("Detected Potholes (Video) | Press 'q' to Exit", hud_frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Status: Video playback finished.")

    def start_live_camera(self) -> None:
        """Launch continuous live webcam stream with real-time detection."""
        self.status_var.set("Starting Live Camera... (Press 'q' on camera window to stop)")
        
        # Run in separate thread to keep Tkinter GUI responsive
        threading.Thread(target=self._live_camera_loop, daemon=True).start()

    def _live_camera_loop(self) -> None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access webcam device index 0.")
            self.status_var.set("Status: Camera failed to open.")
            return

        self.is_camera_running = True
        fps = 0.0
        prev_time = time.perf_counter()

        try:
            while self.is_camera_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                res = self.pipeline.run(frame)

                curr_time = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
                prev_time = curr_time

                hud_frame = draw_hud_overlay(frame, res, fps)
                cv2.imshow("PotholeGuard-AI | Live Camera (Press 'q' to Stop)", hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
        finally:
            self.is_camera_running = False
            cap.release()
            cv2.destroyAllWindows()
            self.status_var.set("Status: Live camera stopped.")

    def _display_side_by_side(self, title1: str, img1: np.ndarray, title2: str, img2: np.ndarray) -> None:
        """Helper to show side-by-side comparison in an OpenCV window."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        target_h = 450
        w1_scaled = int(w1 * (target_h / h1))
        w2_scaled = int(w2 * (target_h / h2))
        
        i1 = cv2.resize(img1, (w1_scaled, target_h))
        i2 = cv2.resize(img2, (w2_scaled, target_h))
        
        # Add title headers
        cv2.putText(i1, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(i2, title2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        
        combined = np.hstack([i1, i2])
        win_title = f"{title1} vs {title2} | Press any key to close"
        cv2.imshow(win_title, combined)
        cv2.waitKey(0)
        cv2.destroyWindow(win_title)
