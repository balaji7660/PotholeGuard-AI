"""
Unit tests for Risk Engine and Safety Refinement Layer.
"""
import pytest
from risk.risk_engine import RiskEngine
from safety.safety_refinement import SafetyRefinementLayer

def test_risk_engine_danger_classification():
    engine = RiskEngine()
    features = {
        "size_score": 0.95,
        "depth_score": 0.95,
        "severity_score": 0.9,
        "path_relevance_score": 0.95,
        "approach_score": 0.95,
        "uncertainty": 0.05,
        "position_class": "CENTER / DIRECT PATH"
    }
    res = engine.evaluate_risk(features)
    assert res["risk_class"] == "DANGER"
    assert res["is_urgent"] is True
    assert len(res["reasons"]) > 0

def test_srl_high_uncertainty_override():
    srl = SafetyRefinementLayer(uncertainty_threshold=0.45)
    
    # Low risk perception but high uncertainty
    detection = {
        "risk_class": "SAFE",
        "risk_score": 0.15,
        "uncertainty": 0.52,
        "depth_class": "SHALLOW",
        "size_class": "SMALL",
        "position_class": "LEFT OF PATH",
        "path_relevance_score": 0.1
    }
    refined = srl.refine_detection(detection)
    assert refined["risk_class"] == "UNCERTAIN"
    assert "UNCERTAINTY_OVERRIDE_SAFE_TO_UNCERTAIN" in refined["srl_overrides"]
