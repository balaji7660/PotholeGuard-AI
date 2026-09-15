"""
FastAPI Backend Server for PotholeGuard-AI.
Real-Time Smartphone Pothole Detection & Rider Safety System.
"""
import os
import io
import time
import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from backend.schemas import (
    HealthResponse, ModelStatusResponse, CalibrationRequest,
    ConfigUpdateRequest, DetectionResponse
)
from backend.database import init_db, log_detection_event, get_recent_history, get_summary_stats
from inference.pipeline import PotholeGuardPipeline

app = FastAPI(
    title="PotholeGuard-AI Backend",
    description="Real-Time Smartphone-Based Pothole Detection & Rider Safety Alert System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
PIPELINE = PotholeGuardPipeline(
    checkpoint_path="checkpoints/best_transunet.pth" if os.path.exists("checkpoints/best_transunet.pth") else None
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "service": "PotholeGuard-AI Backend",
        "version": "1.0.0"
    }

@app.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    return PIPELINE.get_model_status()

@app.get("/api/config")
def get_config():
    return {
        "risk_config": PIPELINE.risk_engine.config,
        "warning_cooldown": PIPELINE.warning_manager.cooldown_seconds,
        "calibration": PIPELINE.size_estimator.calibration
    }

@app.post("/api/config")
def update_config(cfg: ConfigUpdateRequest):
    update_dict = cfg.model_dump(exclude_unset=True)
    if "speech_cooldown_seconds" in update_dict:
        PIPELINE.warning_manager.cooldown_seconds = update_dict.pop("speech_cooldown_seconds")
    PIPELINE.risk_engine.update_config(update_dict)
    return {"status": "SUCCESS", "updated_config": PIPELINE.risk_engine.config}

@app.post("/api/calibrate")
def calibrate_camera(calib: CalibrationRequest):
    calib_dict = calib.model_dump()
    calib_dict["calibrated"] = True
    calib_dict["depth_scale_meters"] = round(calib.camera_height_m / max(0.1, np.sin(np.radians(calib.camera_pitch_deg))), 2)
    PIPELINE.size_estimator.update_calibration(calib_dict)
    PIPELINE.depth_processor.update_calibration(calib_dict)
    return {
        "status": "CALIBRATED",
        "calibration_parameters": calib_dict
    }

@app.get("/api/history")
def history(limit: int = 100):
    return get_recent_history(limit=limit)

@app.get("/api/stats")
def stats():
    return get_summary_stats()

@app.post("/api/test/image")
async def test_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})
    result = PIPELINE.process_frame(frame)
    return result

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    print("[*] Client connected to /ws/detect WebSocket")
    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            
            frame_base64 = data.get("image", "")
            client_ts = data.get("timestamp", None)
            
            if not frame_base64:
                continue

            # Remove prefix if present
            if "," in frame_base64:
                frame_base64 = frame_base64.split(",")[1]

            img_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Process frame through full pipeline
            res = PIPELINE.process_frame(frame, client_timestamp=client_ts)

            # Log to DB if hazard detected
            if res.get("detections"):
                for d in res["detections"]:
                    if d.get("risk_class") in ["WARNING", "DANGER", "UNCERTAIN"]:
                        log_detection_event({
                            "track_id": d.get("track_id", 0),
                            "risk_class": d.get("risk_class"),
                            "risk_score": d.get("risk_score"),
                            "size_class": d.get("size_class"),
                            "width_px": d.get("width_px"),
                            "height_px": d.get("height_px"),
                            "area_px": d.get("area_px"),
                            "relative_depth": d.get("mean_depth", 0.0),
                            "depth_class": d.get("depth_class"),
                            "severity": d.get("severity_score", 0.0),
                            "confidence": d.get("confidence", 0.0),
                            "uncertainty": d.get("uncertainty", 0.0),
                            "path_relevance": d.get("path_relevance_score", 0.0),
                            "recommendation": d.get("recommendation"),
                            "warning_triggered": res.get("voice_alert") is not None,
                            "processing_fps": res.get("processing_fps"),
                            "latency_ms": res.get("latency_ms")
                        })

            await websocket.send_json(res)
    except WebSocketDisconnect:
        print("[-] WebSocket client disconnected")
    except Exception as e:
        print(f"[!] WebSocket error: {e}")

# Mount static frontend
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
