"""
SQLite Database and Logging Layer for PotholeGuard-AI.
Stores detection events, tracks, performance telemetry, and system configuration.
"""
import sqlite3
import json
import time
from typing import List, Dict, Any, Optional

DB_PATH = "potholeguard.db"

def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        track_id INTEGER,
        risk_class TEXT,
        risk_score REAL,
        size_class TEXT,
        width_px INTEGER,
        height_px INTEGER,
        area_px INTEGER,
        relative_depth REAL,
        depth_class TEXT,
        severity REAL,
        confidence REAL,
        uncertainty REAL,
        path_relevance REAL,
        recommendation TEXT,
        warning_triggered INTEGER,
        processing_fps REAL,
        latency_ms REAL,
        latitude REAL,
        longitude REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS system_config (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    conn.commit()
    conn.close()

def log_detection_event(record: Dict[str, Any], db_path: str = DB_PATH):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO detections (
            timestamp, track_id, risk_class, risk_score, size_class,
            width_px, height_px, area_px, relative_depth, depth_class,
            severity, confidence, uncertainty, path_relevance, recommendation,
            warning_triggered, processing_fps, latency_ms, latitude, longitude
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("timestamp", time.time()),
            record.get("track_id", 0),
            record.get("risk_class", "SAFE"),
            record.get("risk_score", 0.0),
            record.get("size_class", "MEDIUM"),
            record.get("width_px", 0),
            record.get("height_px", 0),
            record.get("area_px", 0),
            record.get("relative_depth", 0.0),
            record.get("depth_class", "MEDIUM"),
            record.get("severity", 0.0),
            record.get("confidence", 0.0),
            record.get("uncertainty", 0.0),
            record.get("path_relevance", 0.0),
            record.get("recommendation", "ROAD CLEAR"),
            1 if record.get("warning_triggered", False) else 0,
            record.get("processing_fps", 0.0),
            record.get("latency_ms", 0.0),
            record.get("latitude", None),
            record.get("longitude", None)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] DB Log error: {e}")

def get_recent_history(limit: int = 100, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    res = [dict(r) for r in rows]
    conn.close()
    return res

def get_summary_stats(db_path: str = DB_PATH) -> Dict[str, Any]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM detections")
    total_events = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT track_id) FROM detections WHERE track_id > 0")
    unique_potholes = cur.fetchone()[0]

    cur.execute("SELECT risk_class, COUNT(*) FROM detections GROUP BY risk_class")
    risk_dist = dict(cur.fetchall())

    cur.execute("SELECT COUNT(*) FROM detections WHERE warning_triggered = 1")
    warnings_issued = cur.fetchone()[0]

    cur.execute("SELECT AVG(latency_ms), AVG(processing_fps) FROM detections")
    avg_lat, avg_fps = cur.fetchone()

    conn.close()
    return {
        "total_detection_logs": total_events,
        "unique_potholes_tracked": unique_potholes,
        "risk_distribution": {
            "SAFE": risk_dist.get("SAFE", 0),
            "WARNING": risk_dist.get("WARNING", 0),
            "DANGER": risk_dist.get("DANGER", 0),
            "UNCERTAIN": risk_dist.get("UNCERTAIN", 0)
        },
        "warnings_issued": warnings_issued,
        "average_latency_ms": round(avg_lat or 0.0, 1),
        "average_fps": round(avg_fps or 0.0, 1)
    }

init_db()
