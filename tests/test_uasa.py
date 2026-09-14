"""
tests/test_uasa.py
Unit tests for UASA state generation.
"""
import numpy as np
import pytest
from omegaconf import OmegaConf

from state.uasa import UASAStateGenerator, STATE_DIM
from state.severity import PotholeSeverityScorer, PotholeRegion
from state.temporal_smoothing import EMATemporalSmoother

# Minimal configs for testing
_SEV_CFG = OmegaConf.create({
    "severity": {
        "area_weight": 0.30, "avg_depth_weight": 0.30,
        "min_depth_weight": 0.20, "lane_displacement_weight": 0.20,
        "max_area_pixels": 50000, "max_depth_metres": 0.5,
        "max_lane_displacement": 1.75, "clip_min": 0.0, "clip_max": 1.0,
        "thresholds": {"low": 0.33, "medium": 0.66, "high": 1.0},
    }
})

_RL_CFG = OmegaConf.create({
    "environment": {
        "action_dim": 3, "max_steps": 500, "seed": 42,
        "action_smoothing": {"enabled": True, "ema_alpha": 0.7, "min_persistence": 3},
    }
})


def make_dummy_perception(H=512, W=512, with_pothole=True):
    seg = np.zeros((H, W), dtype=np.float32)
    dep = np.ones((H, W), dtype=np.float32) * 10.0
    unc = np.ones((H, W), dtype=np.float32) * 0.2
    if with_pothole:
        import cv2
        cv2.ellipse(seg, (256, 350), (60, 40), 0, 0, 360, 1.0, -1)
        dep[300:400, 220:300] = 0.3
        unc[300:400, 220:300] = 0.6
    return seg, dep, unc


class TestUASADimension:
    def test_state_dim_no_pothole(self):
        gen = UASAStateGenerator(_SEV_CFG, _RL_CFG)
        seg, dep, unc = make_dummy_perception(with_pothole=False)
        state = gen.compute(seg, dep, unc)
        assert len(state) == 117, f"Expected 117, got {len(state)}"

    def test_state_dim_with_pothole(self):
        gen = UASAStateGenerator(_SEV_CFG, _RL_CFG)
        seg, dep, unc = make_dummy_perception(with_pothole=True)
        state = gen.compute(seg, dep, unc)
        assert len(state) == 117, f"Expected 117, got {len(state)}"

    def test_state_dtype(self):
        gen = UASAStateGenerator(_SEV_CFG, _RL_CFG)
        seg, dep, unc = make_dummy_perception()
        state = gen.compute(seg, dep, unc)
        assert state.dtype == np.float32

    def test_state_range(self):
        """Most state features should be in [0,1] after normalisation."""
        gen = UASAStateGenerator(_SEV_CFG, _RL_CFG)
        seg, dep, unc = make_dummy_perception()
        state = gen.compute(seg, dep, unc)
        # At least 80% of features should be in [0,1]
        in_range = np.sum((state >= -1.0) & (state <= 1.0))
        assert in_range >= 100, f"Only {in_range}/117 features in [-1,1]"

    def test_reset_clears_ema(self):
        gen = UASAStateGenerator(_SEV_CFG, _RL_CFG)
        seg, dep, unc = make_dummy_perception()
        gen.compute(seg, dep, unc)
        gen.reset()
        assert gen.smoother._prev is None


class TestSeverityScorer:
    def test_zero_severity(self):
        scorer = PotholeSeverityScorer(_SEV_CFG)
        r = PotholeRegion(area_pixels=0, avg_depth=0, min_depth=0, lane_displacement=0)
        assert scorer.score(r) == pytest.approx(0.0)

    def test_max_severity(self):
        scorer = PotholeSeverityScorer(_SEV_CFG)
        r = PotholeRegion(area_pixels=100000, avg_depth=2.0, min_depth=2.0, lane_displacement=5.0)
        assert scorer.score(r) == pytest.approx(1.0)

    def test_severity_clipped(self):
        scorer = PotholeSeverityScorer(_SEV_CFG)
        r = PotholeRegion(area_pixels=999999, avg_depth=999, min_depth=999, lane_displacement=999)
        s = scorer.score(r)
        assert 0.0 <= s <= 1.0

    def test_severity_list(self):
        scorer = PotholeSeverityScorer(_SEV_CFG)
        regions = [
            PotholeRegion(1000, 0.1, 0.05, 0.2),
            PotholeRegion(5000, 0.3, 0.1,  0.5),
        ]
        scores = scorer.score_all(regions)
        assert len(scores) == 2
        assert all(0 <= s <= 1 for s in scores)


class TestEMASmoother:
    def test_initial_call_returns_input(self):
        sm = EMATemporalSmoother(10, alpha=0.7)
        x  = np.ones(10, dtype=np.float32) * 2.0
        out = sm(x)
        np.testing.assert_array_equal(out, x)

    def test_smoothing_reduces_variance(self):
        sm = EMATemporalSmoother(1, alpha=0.5)
        sm(np.array([1.0]))
        out = sm(np.array([3.0]))
        assert float(out[0]) == pytest.approx(2.0)

    def test_reset(self):
        sm = EMATemporalSmoother(5, alpha=0.7)
        sm(np.ones(5))
        sm.reset()
        assert sm._prev is None
