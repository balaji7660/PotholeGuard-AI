"""
Unit tests for vision feature extractors.
"""
import numpy as np
import pytest
from vision.size_estimator import SizeEstimator
from vision.depth_processor import DepthProcessor
from vision.uncertainty_processor import UncertaintyProcessor
from vision.path_estimator import PathEstimator

def test_size_estimator():
    estimator = SizeEstimator()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:260, 280:360] = 255
    bbox = (280, 200, 360, 260)
    res = estimator.estimate_size(mask, bbox, (480, 640))
    
    assert res["size_class"] in ["SMALL", "MEDIUM", "LARGE", "VERY LARGE"]
    assert res["width_px"] == 80
    assert res["height_px"] == 60
    assert res["area_px"] == 4800

def test_depth_processor():
    processor = DepthProcessor()
    depth_map = np.ones((480, 640), dtype=np.float32) * 0.85
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:250, 200:250] = 255
    res = processor.process_depth(depth_map, mask)
    
    assert res["depth_class"] == "DEEP"
    assert res["mean_depth"] == 0.85

def test_uncertainty_processor():
    processor = UncertaintyProcessor(high_uncertainty_threshold=0.4)
    unc_map = np.ones((480, 640), dtype=np.float32) * 0.55
    mask = np.ones((480, 640), dtype=np.uint8) * 255
    res = processor.process_uncertainty(unc_map, mask)
    
    assert res["is_high_uncertainty"] is True
    assert res["uncertainty"] == 0.55

def test_path_estimator():
    estimator = PathEstimator()
    # Centered pothole at bottom of frame (direct path)
    res = estimator.estimate_path_relevance((320, 420), (480, 640), (280, 380, 360, 460))
    assert res["position_class"] == "CENTER / DIRECT PATH"
    assert res["relevance_class"] == "HIGH"
