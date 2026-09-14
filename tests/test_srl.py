"""
tests/test_srl.py
Unit tests for the Safety Refinement Layer.
"""
import numpy as np
import pytest
from omegaconf import OmegaConf
from safety.srl import SafetyRefinementLayer

_SRL_CFG = OmegaConf.create({
    "srl": {
        "enabled": True,
        "uncertainty_threshold": 0.75,
        "severity_threshold": 0.80,
        "lateral_deviation_threshold": 0.5,
        "collision_risk_threshold": 0.5,
        "action_consistency_window": 5,
        "oscillation_count_threshold": 3,
        "fallback": {
            "uncertainty_override": 0,
            "severity_override": 0,
            "collision_override": 0,
            "oscillation_override": 0,
        },
        "emergency_brake": {
            "enabled": True,
            "min_depth_threshold": 0.5,
            "severity_threshold": 0.95,
        },
    },
    "collision_checker": {
        "vehicle": {"wheelbase": 2.7, "width": 1.8, "length": 4.5},
        "horizon_steps": 5, "step_size": 0.5,
        "action_lateral_offsets": {0: 0.0, 1: -0.5, 2: 0.5},
        "lane_half_width": 1.75,
    },
})


def empty_mask(): return np.zeros((512, 512), dtype=np.uint8)
def full_mask():  return np.ones((512, 512), dtype=np.uint8)


class TestSRL:
    def test_accepts_safe_action(self):
        srl = SafetyRefinementLayer(_SRL_CFG)
        dec = srl.evaluate(0, 0.1, 0.1, 0.0, 10.0, empty_mask())
        assert dec.accepted is True
        assert dec.final_action == 0

    def test_overrides_high_uncertainty(self):
        srl = SafetyRefinementLayer(_SRL_CFG)
        dec = srl.evaluate(1, 0.9, 0.1, 0.0, 10.0, empty_mask())
        assert dec.accepted is False
        assert "uncertainty" in dec.override_reason.lower()

    def test_overrides_extreme_severity(self):
        srl = SafetyRefinementLayer(_SRL_CFG)
        dec = srl.evaluate(0, 0.1, 0.95, 0.0, 10.0, empty_mask())
        assert dec.accepted is False

    def test_emergency_brake_trigger(self):
        srl = SafetyRefinementLayer(_SRL_CFG)
        dec = srl.evaluate(0, 0.2, 0.97, 0.0, 0.3, empty_mask())
        assert dec.accepted is False
        assert "EMERGENCY" in dec.override_reason

    def test_intervention_rate(self):
        srl = SafetyRefinementLayer(_SRL_CFG)
        srl.evaluate(0, 0.1, 0.1, 0.0, 10.0, empty_mask())   # safe
        srl.evaluate(1, 0.9, 0.1, 0.0, 10.0, empty_mask())   # override
        srl.evaluate(0, 0.1, 0.1, 0.0, 10.0, empty_mask())   # safe
        assert abs(srl.intervention_rate - 1/3) < 0.01

    def test_scenario_a_accepts(self):
        srl = SafetyRefinementLayer(_SRL_CFG)
        dec = srl.scenario_a(empty_mask())
        assert dec.accepted is True

    def test_scenario_d_emergency(self):
        srl = SafetyRefinementLayer(_SRL_CFG)
        dec = srl.scenario_d(empty_mask())
        assert dec.accepted is False
        assert dec.override_reason is not None
