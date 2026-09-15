"""
Integration tests for FastAPI Backend endpoints.
"""
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "PotholeGuard" in data["service"]

def test_model_status_endpoint():
    res = client.get("/model/status")
    assert res.status_code == 200
    data = res.json()
    assert "mode" in data
    assert data["model_name"] == "Multi-Task TransUNet"

def test_config_endpoints():
    res_get = client.get("/api/config")
    assert res_get.status_code == 200
    
    res_post = client.post("/api/config", json={"danger_threshold": 0.70})
    assert res_post.status_code == 200
    assert res_post.json()["status"] == "SUCCESS"

def test_calibration_endpoint():
    payload = {
        "camera_height_m": 1.1,
        "camera_pitch_deg": 18.0,
        "reference_distance_m": 4.5,
        "pixel_to_cm_ratio": 0.12
    }
    res = client.post("/api/calibrate", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "CALIBRATED"

def test_history_and_stats_endpoints():
    res_hist = client.get("/api/history")
    assert res_hist.status_code == 200
    assert isinstance(res_hist.json(), list)

    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    assert "total_detection_logs" in res_stats.json()
