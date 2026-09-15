"""
gui/app.py
High-Performance Tkinter GUI for Pothole Detection & Avoidance System.
Dual-pane layout with integrated live video display canvas and non-blocking pipeline stages.
"""
from __future__ import annotations

import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

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
        self.root.title("Pothole Detection & Avoidance System")
        self.root.geometry("1020x620")
        self.root.minsize(900, 560)
        self.root.configure(bg="#0f172a")

        # Initialize pipeline instance
        self.pipeline = InferencePipeline(config_dir=str(ROOT / "configs"))

        # State variables
        self.current_image_path: Optional[str] = None
        self.current_video_path: Optional[str] = None
        self.current_image_bgr: Optional[np.ndarray] = None
        self.is_camera_running = False
        self.is_video_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_cap: Optional[cv2.VideoCapture] = None
        
        self.prev_time = time.perf_counter()
        self.fps = 0.0

        # Build UI layout
        self._build_ui()
        
        # Load default demo image on startup so buttons work immediately
        self._load_default_image()

        # Handle window close cleanup
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        # Header title bar
        header = tk.Frame(self.root, bg="#1e293b", pady=10, padx=16)
        header.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="🚗 POTHOLEGUARD-AI : DETECTION & AVOIDANCE SYSTEM",
            font=("Helvetica", 14, "bold"),
            bg="#1e293b",
            fg="#38bdf8"
        )
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(
            header,
            text="Digital Image Processing & Multi-Task Perception",
            font=("Helvetica", 9),
            bg="#1e293b",
            fg="#94a3b8"
        )
        sub_lbl.pack(side=tk.RIGHT)

        # Main body container (Dual Pane)
        main_body = tk.Frame(self.root, bg="#0f172a", padx=14, pady=12)
        main_body.pack(fill=tk.BOTH, expand=True)

        # ── Left Control Panel ──
        left_panel = tk.Frame(main_body, bg="#1e293b", width=280, padx=14, pady=14, relief=tk.FLAT)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_panel.pack_propagate(False)

        panel_title = tk.Label(
            left_panel,
            text="⚙️ PIPELINE STAGES",
            font=("Helvetica", 10, "bold"),
            bg="#1e293b",
            fg="#f1f5f9"
        )
        panel_title.pack(anchor="w", pady=(0, 12))

        # 8 Functional Pipeline Buttons
        self.btn_upload = self._create_button(left_panel, "📁 Upload Image/Video", self.upload_file, "#334155", "#ffffff")
        self.btn_enhance = self._create_button(left_panel, "🔆 Enhancement", self.show_enhancement, "#334155", "#ffffff")
        self.btn_restore = self._create_button(left_panel, "🧹 Restoration", self.show_restoration, "#334155", "#ffffff")
        self.btn_morph = self._create_button(left_panel, "🔬 Morphological Processing", self.show_morphology, "#334155", "#ffffff")
        self.btn_seg = self._create_button(left_panel, "📐 Segmentation", self.show_segmentation, "#334155", "#ffffff")
        self.btn_det_img = self._create_button(left_panel, "🎯 Detected Potholes (Image)", self.show_detected_image, "#334155", "#ffffff")
        self.btn_det_vid = self._create_button(left_panel, "🎬 View Detected Potholes (Video)", self.toggle_video_detection, "#334155", "#ffffff")
        self.btn_live_cam = self._create_button(left_panel, "🎥 Start Live Camera", self.toggle_live_camera, "#0284c7", "#ffffff", is_primary=True)

        # ── Right Visualizer Panel ──
        right_panel = tk.Frame(main_body, bg="#0f172a")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Top Action & Status Indicator Banner
        self.action_banner = tk.Label(
            right_panel,
            text="⬆️ MAINTAIN LANE — ROAD SURFACE CLEAR",
            font=("Helvetica", 11, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            pady=8
        )
        self.action_banner.pack(fill=tk.X, pady=(0, 8))

        # Display Canvas / Label
        self.display_lbl = tk.Label(right_panel, bg="#000000", bd=1, relief=tk.SOLID)
        self.display_lbl.pack(fill=tk.BOTH, expand=True)

        # Bottom Telemetry Bar
        self.telemetry_bar = tk.Label(
            right_panel,
            text="FPS: -- | Latency: -- ms | Potholes Detected: 0 | Severity: 0.00",
            font=("Helvetica", 9),
            bg="#1e293b",
            fg="#94a3b8",
            pady=6,
            anchor="w",
            padx=10
        )
        self.telemetry_bar.pack(fill=tk.X, pady=(8, 0))

        # Status footer bar
        self.status_var = tk.StringVar(value="Ready. Click any pipeline button or start live camera.")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 8),
            bg="#090d16",
            fg="#64748b",
            anchor="w",
            padx=12,
            pady=4
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_button(self, parent: tk.Widget, text: str, cmd, bg: str, fg: str, is_primary: bool = False) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=cmd,
            font=("Helvetica", 9, "bold" if is_primary else "normal"),
            bg=bg,
            fg=fg,
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            pady=7,
            bd=0
        )
        btn.pack(fill=tk.X, pady=4)
        return btn

    def _load_default_image(self) -> None:
        """Load default demo image so buttons work immediately."""
        try:
            from app.demo_images import get_demo_image
            self.current_image_bgr = get_demo_image("center_pothole")
            self._render_frame(self.current_image_bgr)
            self.status_var.set("Demo road image loaded. Click Enhancement, Segmentation, or Start Live Camera.")
        except Exception:
            pass

    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        """Render OpenCV BGR frame smoothly onto the Tkinter display label."""
        if frame_bgr is None:
            return

        # Resize to fit display label area while keeping aspect ratio
        canvas_w = max(400, self.display_lbl.winfo_width())
        canvas_h = max(300, self.display_lbl.winfo_height())
        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = 640, 480

        h, w = frame_bgr.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame_bgr, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.display_lbl.img_tk = img_tk
        self.display_lbl.configure(image=img_tk)

    def _update_hud_banner(self, res: PipelineResult, fps: float) -> None:
        """Update top action banner and telemetry."""
        action = res.final_action
        action_name = ACTION_NAMES.get(action, "Maintain Lane").upper()
        is_override = (res.srl_decision is not None and not res.srl_decision.accepted)
        is_brake = (res.srl_decision and res.srl_decision.override_reason and "EMERGENCY" in res.srl_decision.override_reason)

        if is_brake:
            self.action_banner.config(text="🛑 EMERGENCY BRAKE — CRITICAL HAZARD AHEAD", bg="#dc2626")
        elif action == 1:
            tag = "SRL OVERRIDE" if is_override else "RL AVOIDANCE"
            self.action_banner.config(text=f"⬅️ SHIFT LEFT ({tag})", bg="#16a34a")
        elif action == 2:
            tag = "SRL OVERRIDE" if is_override else "RL AVOIDANCE"
            self.action_banner.config(text=f"➡️ SHIFT RIGHT ({tag})", bg="#16a34a")
        else:
            self.action_banner.config(text="⬆️ MAINTAIN LANE — ROAD TRAJECTORY CLEAR", bg="#0284c7")

        severity = float(res.state_vector[6]) if len(res.state_vector) > 6 else 0.0
        self.telemetry_bar.config(
            text=f"FPS: {fps:.1f} | Latency: {res.inference_ms:.1f} ms | Potholes Detected: {res.n_potholes} | Severity: {severity:.2f} | Mode: {'Demo' if res.demo_mode else 'Model'}"
        )

    # ── Pipeline Button Actions ────────────────────────────────────────────────

    def upload_file(self) -> None:
        """Upload image or video file."""
        self._stop_all_streams()
        filetypes = [
            ("Supported Media", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
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
            self._render_frame(self.current_image_bgr)
            self.status_var.set(f"Loaded Image: {Path(filepath).name}")
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            self.current_video_path = filepath
            self.current_image_path = None
            self.status_var.set(f"Loaded Video: {Path(filepath).name}. Click 'View Detected Potholes (Video)'.")
            messagebox.showinfo("Video Loaded", f"Loaded: {Path(filepath).name}\nClick 'View Detected Potholes (Video)' to start detection.")

    def show_enhancement(self) -> None:
        self._stop_all_streams()
        if self.current_image_bgr is None: return
        self.status_var.set("Running Enhancement (CLAHE)...")
        enhanced = enhance_image(self.current_image_bgr)
        self._render_frame(enhanced)
        self.action_banner.config(text="🔆 STAGE: CONTRAST ENHANCEMENT (CLAHE)", bg="#0284c7")
        self.status_var.set("Stage 1/4: Image Contrast Enhancement Complete.")

    def show_restoration(self) -> None:
        self._stop_all_streams()
        if self.current_image_bgr is None: return
        self.status_var.set("Running Restoration (Bilateral Denoising)...")
        restored = restore_image(self.current_image_bgr)
        self._render_frame(restored)
        self.action_banner.config(text="🧹 STAGE: NOISE RESTORATION & BILATERAL FILTERING", bg="#0284c7")
        self.status_var.set("Stage 2/4: Noise Restoration Complete.")

    def show_morphology(self) -> None:
        self._stop_all_streams()
        if self.current_image_bgr is None: return
        self.status_var.set("Running Morphological Operations...")
        morph = morphological_processing(self.current_image_bgr)
        self._render_frame(morph)
        self.action_banner.config(text="🔬 STAGE: MORPHOLOGICAL OPERATIONS (OPENING / CLOSING)", bg="#0284c7")
        self.status_var.set("Stage 3/4: Morphological Structure Isolation Complete.")

    def show_segmentation(self) -> None:
        self._stop_all_streams()
        if self.current_image_bgr is None: return
        self.status_var.set("Running Pothole Segmentation...")
        seg = segment_potholes(self.current_image_bgr)
        self._render_frame(seg)
        self.action_banner.config(text="📐 STAGE: POTHOLE REGION SEGMENTATION & CONTOURS", bg="#0284c7")
        self.status_var.set("Stage 4/4: Pothole Region Segmentation Complete.")

    def show_detected_image(self) -> None:
        self._stop_all_streams()
        if self.current_image_bgr is None: return
        self.status_var.set("Running AI Perception & Steering Avoidance...")
        
        t0 = time.perf_counter()
        res = self.pipeline.run(self.current_image_bgr)
        fps = 1.0 / max(1e-4, time.perf_counter() - t0)
        
        detected_vis = draw_hud_overlay(self.current_image_bgr, res, fps)
        self._render_frame(detected_vis)
        self._update_hud_banner(res, fps)
        self.status_var.set(f"Detection Done: {res.n_potholes} potholes detected. Action: {ACTION_NAMES.get(res.final_action)}.")

    def toggle_video_detection(self) -> None:
        if self.is_video_running:
            self._stop_all_streams()
            return

        if not self.current_video_path:
            messagebox.showwarning("No Video", "Please click 'Upload Image/Video' and select a video file first.")
            return

        self._stop_all_streams()
        self.video_cap = cv2.VideoCapture(self.current_video_path)
        if not self.video_cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

        self.is_video_running = True
        self.btn_det_vid.config(text="⏹️ Stop Video", bg="#dc2626")
        self.status_var.set(f"Playing video: {Path(self.current_video_path).name}")
        self._process_video_frame()

    def _process_video_frame(self) -> None:
        if not self.is_video_running or self.video_cap is None:
            return

        ret, frame = self.video_cap.read()
        if not ret or frame is None:
            self._stop_all_streams()
            self.status_var.set("Video playback completed.")
            return

        t_curr = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-4, t_curr - self.prev_time))
        self.prev_time = t_curr

        res = self.pipeline.run(frame)
        hud_frame = draw_hud_overlay(frame, res, self.fps)
        self._render_frame(hud_frame)
        self._update_hud_banner(res, self.fps)

        self.root.after(25, self._process_video_frame)

    def toggle_live_camera(self) -> None:
        if self.is_camera_running:
            self._stop_all_streams()
            return

        self._stop_all_streams()
        # Open camera with Windows DirectShow backend
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open Webcam 0. Please verify your camera is connected and not locked by another app.")
            self.status_var.set("Status: Camera failed to open.")
            return

        self.is_camera_running = True
        self.btn_live_cam.config(text="⏹️ Stop Live Camera", bg="#dc2626")
        self.status_var.set("Live Camera Streaming Active (30 FPS).")
        self._process_camera_frame()

    def _process_camera_frame(self) -> None:
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._stop_all_streams()
            self.status_var.set("Failed to capture frame from webcam.")
            return

        t_curr = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-4, t_curr - self.prev_time))
        self.prev_time = t_curr

        # Run pipeline
        res = self.pipeline.run(frame)
        hud_frame = draw_hud_overlay(frame, res, self.fps)
        
        self._render_frame(hud_frame)
        self._update_hud_banner(res, self.fps)

        # Loop smoothly using Tkinter after()
        self.root.after(20, self._process_camera_frame)

    def _stop_all_streams(self) -> None:
        """Stop any active camera or video streams."""
        self.is_camera_running = False
        self.is_video_running = False
        self.btn_live_cam.config(text="🎥 Start Live Camera", bg="#0284c7")
        self.btn_det_vid.config(text="🎬 View Detected Potholes (Video)", bg="#334155")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

    def _on_close(self) -> None:
        """Cleanly release resources on exit."""
        self._stop_all_streams()
        self.root.destroy()
