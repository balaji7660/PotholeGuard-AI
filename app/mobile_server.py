"""
app/mobile_server.py
High-Performance Real-Time Mobile Server for PotholeGuard-AI.

Provides:
- Low-latency WebSocket endpoint for mobile camera video streaming
- REST endpoints for frame inference
- AR overlay contour generation & telemetry serialization
- Multi-modal view data (Depth, Uncertainty, Simulation)
- Dynamic voice alert synthesis prompts
- Responsive Mobile HUD interface serving
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import matplotlib.cm as cm
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import InferencePipeline, ACTION_NAMES, PipelineResult

app = FastAPI(
    title="PotholeGuard-AI Mobile Real-Time Server",
    description="Real-time pothole perception, RL ensemble decision & SRL safety system for mobile testing",
    version="1.0.0"
)

# Enable CORS for mobile devices connecting over LAN/tunnels
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global pipeline instance
_pipeline: Optional[InferencePipeline] = None

def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline(config_dir=str(ROOT / "configs"))
    return _pipeline


def colorize_depth_b64(depth: np.ndarray, quality: int = 65) -> str:
    """Colorize depth map with plasma colormap and encode to JPEG Base64."""
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    colored_rgb = (cm.plasma(d_norm)[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", colored_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode("utf-8")


def colorize_uncertainty_b64(unc: np.ndarray, quality: int = 65) -> str:
    """Colorize uncertainty map with hot colormap and encode to JPEG Base64."""
    u_norm = np.clip(unc, 0, 1)
    colored_rgb = (cm.hot(u_norm)[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", colored_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode("utf-8")


def image_to_b64(img_bgr: np.ndarray, quality: int = 70) -> str:
    """Encode BGR image to JPEG Base64."""
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode("utf-8")


def extract_contours_and_boxes(seg_mask: np.ndarray, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """
    Extract vector contours and normalized bounding boxes from binary segmentation mask.
    This allows client-side AR rendering without transmitting heavy overlay images.
    """
    binary = (seg_mask >= 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    mask_h, mask_w = seg_mask.shape[:2]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # ignore tiny noise specks
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        # Approximate contour polygon to reduce payload size
        epsilon = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Normalize coordinates [0, 1] relative to mask dimensions
        norm_pts = [[float(pt[0][0]) / mask_w, float(pt[0][1]) / mask_h] for pt in approx]
        
        results.append({
            "box": {
                "x": float(x) / mask_w,
                "y": float(y) / mask_h,
                "w": float(w) / mask_w,
                "h": float(h) / mask_h,
            },
            "polygon": norm_pts,
            "area_px": int(area),
            "area_ratio": float(area) / (mask_w * mask_h)
        })
    return results


def build_voice_prompt(res: PipelineResult, prev_action: Optional[int] = None) -> Optional[str]:
    """Generate concise voice warning when hazards or lane changes are required."""
    final_act = res.final_action
    srl = res.srl_decision
    
    # Priority 1: Emergency Brake
    if srl is not None and srl.override_reason and "EMERGENCY" in srl.override_reason:
        return "Emergency brake! Severe hazard ahead."
        
    # Priority 2: SRL Override
    if srl is not None and not srl.accepted:
        if final_act == 1:
            return "Safety override. Steer left."
        elif final_act == 2:
            return "Safety override. Steer right."
            
    # Priority 3: Action Change with significant pothole detection
    if res.n_potholes > 0 and final_act != prev_action:
        severity = float(res.state_vector[6]) if len(res.state_vector) > 6 else 0.0
        if severity > 0.4:
            if final_act == 1:
                return "Pothole detected. Shift left."
            elif final_act == 2:
                return "Pothole detected. Shift right."
            elif final_act == 0 and severity > 0.7:
                return "Caution. Rough road ahead."
                
    return None


def process_frame_internal(image_bgr: np.ndarray, include_maps: bool = False, prev_action: Optional[int] = None) -> Dict[str, Any]:
    """Run pipeline and format complete JSON payload for mobile client."""
    t_start = time.perf_counter()
    pipeline = get_pipeline()
    
    h_orig, w_orig = image_bgr.shape[:2]
    res: PipelineResult = pipeline.run(image_bgr)
    
    severity_val = float(res.state_vector[6]) if len(res.state_vector) > 6 else 0.0
    if severity_val < 0.33:
        sev_label = "LOW"
    elif severity_val < 0.66:
        sev_label = "MEDIUM"
    else:
        sev_label = "HIGH"
        
    uncertainty_mean = float(np.mean(res.uncertainty))
    
    # Vector contours for smooth client-side canvas rendering
    pothole_items = extract_contours_and_boxes(res.segmentation, w_orig, h_orig)
    
    voice_msg = build_voice_prompt(res, prev_action)
    
    # Base response payload
    payload: Dict[str, Any] = {
        "status": "success",
        "demo_mode": res.demo_mode,
        "n_potholes": len(pothole_items),
        "potholes": pothole_items,
        "action": int(res.final_action),
        "action_name": ACTION_NAMES.get(res.final_action, "Maintain Lane"),
        "rl_action": int(res.rl_action),
        "rl_action_name": ACTION_NAMES.get(res.rl_action, "Maintain Lane"),
        "srl_override": bool(res.srl_decision is not None and not res.srl_decision.accepted),
        "override_reason": res.srl_decision.override_reason if res.srl_decision else None,
        "severity": round(severity_val, 3),
        "severity_level": sev_label,
        "uncertainty": round(uncertainty_mean, 3),
        "vehicle_x": round(float(pipeline.vehicle.x_norm), 3),
        "ensemble_probs": [round(float(p), 3) for p in res.ensemble_probs],
        "agent_probs": {k: [round(float(v), 3) for v in probs] for k, probs in res.agent_probs.items()},
        "voice_alert": voice_msg,
        "inference_ms": round(res.inference_ms, 1),
        "total_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
    }
    
    # Include Base64 heatmaps when requested
    if include_maps:
        payload["depth_b64"] = colorize_depth_b64(res.depth)
        payload["uncertainty_b64"] = colorize_uncertainty_b64(res.uncertainty)
        if res.road_image is not None:
            payload["road_sim_b64"] = image_to_b64(res.road_image)
            
    return payload


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_mobile_hud():
    """Serve the primary Mobile HUD HTML interface."""
    hud_file = STATIC_DIR / "mobile_hud.html"
    if hud_file.exists():
        return FileResponse(str(hud_file))
    return HTMLResponse("<h1>Mobile HUD not found. Please verify app/static/mobile_hud.html exists.</h1>", status_code=404)


@app.get("/api/health")
async def health_check():
    """Health status and configuration."""
    pipeline = get_pipeline()
    return {
        "status": "online",
        "demo_mode": pipeline.demo_mode,
        "timestamp": time.time(),
        "device": "CPU" if pipeline._perception_model is None else "Model Active"
    }


class FrameData(BaseModel):
    image_b64: str
    include_maps: bool = False
    prev_action: Optional[int] = None


@app.post("/api/detect_frame")
async def detect_frame_json(data: FrameData):
    """REST endpoint for single frame detection via base64 encoded image."""
    try:
        header_split = data.image_b64.split(",")
        raw_b64 = header_split[1] if len(header_split) > 1 else header_split[0]
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse({"status": "error", "message": "Failed to decode image"}, status_code=400)
            
        result = process_frame_internal(img_bgr, include_maps=data.include_maps, prev_action=data.prev_action)
        return result
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/upload_frame")
async def upload_frame_file(
    file: UploadFile = File(...),
    include_maps: bool = Form(False),
    prev_action: Optional[int] = Form(None)
):
    """REST endpoint for image file upload."""
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse({"status": "error", "message": "Failed to decode uploaded image"}, status_code=400)
            
        result = process_frame_internal(img_bgr, include_maps=include_maps, prev_action=prev_action)
        return result
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    High-speed bi-directional WebSocket stream for mobile real-time video feed.
    Receives base64/binary frames from mobile camera and streams back instant detection overlays & decisions.
    """
    await websocket.accept()
    prev_action = None
    frame_count = 0
    
    try:
        while True:
            # Receive message from mobile client
            message = await websocket.receive_text()
            data = json.loads(message)
            
            cmd = data.get("cmd", "frame")
            if cmd == "ping":
                await websocket.send_text(json.dumps({"cmd": "pong", "time": time.time()}))
                continue
                
            img_b64 = data.get("image", "")
            if not img_b64:
                continue
                
            include_maps = bool(data.get("include_maps", False))
            
            # Decode frame
            header_split = img_b64.split(",")
            raw_b64 = header_split[1] if len(header_split) > 1 else header_split[0]
            img_bytes = base64.b64decode(raw_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img_bgr is not None:
                result = process_frame_internal(img_bgr, include_maps=include_maps, prev_action=prev_action)
                result["frame_id"] = data.get("frame_id", frame_count)
                prev_action = result["action"]
                frame_count += 1
                await websocket.send_text(json.dumps(result))
            else:
                await websocket.send_text(json.dumps({"status": "error", "message": "Bad frame"}))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except Exception:
            pass
