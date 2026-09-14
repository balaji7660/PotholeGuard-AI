"""
tests/test_reward.py
Unit tests for the reward function.
"""
import pytest
from omegaconf import OmegaConf
from rl.reward import RewardCalculator

_RL_CFG = OmegaConf.create({
    "reward": {
        "progress_reward": 1.0,
        "severity_penalty": 2.0,
        "uncertainty_penalty": 0.5,
        "steering_penalty": 0.3,
        "collision_penalty": 10.0,
        "high_severity_threshold": 0.6,
        "high_uncertainty_threshold": 0.7,
    }
})


class TestRewardCalculator:
    def test_clean_step_positive(self):
        calc = RewardCalculator(_RL_CFG)
        r = calc.compute(0.0, 0.0, 0, 0, False)
        assert r.total == pytest.approx(1.0)

    def test_collision_large_negative(self):
        calc = RewardCalculator(_RL_CFG)
        r = calc.compute(0.0, 0.0, 0, 0, True)
        assert r.total < 0
        assert r.collision == pytest.approx(-10.0)

    def test_steering_penalty_on_change(self):
        calc = RewardCalculator(_RL_CFG)
        r = calc.compute(0.0, 0.0, 1, 0, False)  # action changed 0→1
        assert r.steering == pytest.approx(-0.3)

    def test_no_steering_penalty_same(self):
        calc = RewardCalculator(_RL_CFG)
        r = calc.compute(0.0, 0.0, 1, 1, False)  # same action
        assert r.steering == pytest.approx(0.0)

    def test_severity_penalty_above_threshold(self):
        calc = RewardCalculator(_RL_CFG)
        r = calc.compute(0.9, 0.0, 0, 0, False)
        assert r.severity < 0

    def test_no_severity_penalty_below_threshold(self):
        calc = RewardCalculator(_RL_CFG)
        r = calc.compute(0.3, 0.0, 0, 0, False)
        assert r.severity == pytest.approx(0.0)

    def test_total_components(self):
        calc = RewardCalculator(_RL_CFG)
        r = calc.compute(0.5, 0.5, 1, 0, False)
        assert abs(r.total - (r.progress + r.severity + r.uncertainty + r.steering + r.collision)) < 1e-5
