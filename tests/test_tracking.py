"""
Unit tests for Temporal Tracker and Warning Manager.
"""
import pytest
from tracking.tracker import PotholeTracker
from risk.warning_manager import WarningManager

def test_tracker_association():
    tracker = PotholeTracker()
    
    # Frame 1
    det_f1 = [{
        "bbox": (100, 100, 150, 150),
        "size_score": 0.5,
        "depth_score": 0.5,
        "uncertainty": 0.1,
        "confidence": 0.9,
        "path_relevance_score": 0.5,
        "risk_score": 0.5,
        "risk_class": "WARNING"
    }]
    tracks_f1 = tracker.update(det_f1)
    assert len(tracks_f1) == 1
    t1_id = tracks_f1[0].track_id

    # Frame 2: slightly shifted box
    det_f2 = [{
        "bbox": (102, 103, 152, 153),
        "size_score": 0.55,
        "depth_score": 0.55,
        "uncertainty": 0.1,
        "confidence": 0.9,
        "path_relevance_score": 0.5,
        "risk_score": 0.55,
        "risk_class": "WARNING"
    }]
    tracks_f2 = tracker.update(det_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t1_id
    assert tracks_f2[0].hits == 2

def test_warning_deduplication():
    wm = WarningManager(cooldown_seconds=1.0)
    tracker = PotholeTracker()
    
    det = [{
        "bbox": (100, 100, 200, 200),
        "size_score": 0.9,
        "depth_score": 0.9,
        "uncertainty": 0.05,
        "confidence": 0.95,
        "path_relevance_score": 0.8,
        "risk_score": 0.85,
        "risk_class": "DANGER",
        "size_class": "LARGE",
        "depth_class": "DEEP"
    }]
    
    tracks = tracker.update(det)
    alert1 = wm.process_tracks_for_warnings(tracks)
    assert alert1 is not None
    assert alert1["alert_type"] == "DANGER"

    # Next immediate check should be throttled (no duplicate voice spam)
    alert2 = wm.process_tracks_for_warnings(tracks)
    assert alert2 is None
