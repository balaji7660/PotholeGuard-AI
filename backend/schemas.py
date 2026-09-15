"""
Pydantic Schemas for PotholeGuard-AI Backend.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class ModelStatusResponse(BaseModel):
    model_name: str
    weights_loaded: bool
    mode: str
    checkpoint: str
    device: str
    status_message: str

class CalibrationRequest(BaseModel):
    camera_height_m: float = Field(default=1.0, description="Smartphone mount height from road in meters")
    camera_pitch_deg: float = Field(default=15.0, description="Smartphone camera downward tilt angle in degrees")
    reference_distance_m: float = Field(default=5.0, description="Known distance to calibration marker")
    pixel_to_cm_ratio: float = Field(default=0.15, description="Calibrated pixel to cm scale factor")

class ConfigUpdateRequest(BaseModel):
    size_weight: Optional[float] = 0.25
    depth_weight: Optional[float] = 0.30
    severity_weight: Optional[float] = 0.15
    path_weight: Optional[float] = 0.20
    uncertainty_weight: Optional[float] = 0.10
    warning_threshold: Optional[float] = 0.35
    danger_threshold: Optional[float] = 0.65
    uncertainty_threshold: Optional[float] = 0.45
    speech_cooldown_seconds: Optional[float] = 3.5

class PotholeDetectionItem(BaseModel):
    track_id: int
    bbox: List[int]
    area_px: int
    width_px: int
    height_px: int
    size_class: str
    relative_depth: float
    depth_class: str
    severity: float
    confidence: float
    uncertainty: float
    path_relevance: float
    risk_score: float
    risk_class: str
    recommendation: str

class DetectionResponse(BaseModel):
    timestamp: float
    frame_id: int
    mode: str
    processing_fps: float
    latency_ms: float
    network_latency_ms: float
    overall_risk: str
    overall_recommendation: str
    voice_alert: Optional[Dict[str, Any]] = None
    detections: List[Dict[str, Any]]
