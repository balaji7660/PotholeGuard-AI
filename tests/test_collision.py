"""
tests/test_collision.py
Unit tests for collision checker and trajectory predictor.
"""
import numpy as np
import pytest
from omegaconf import OmegaConf
from safety.collision_checker import CollisionChecker
from safety.trajectory import TrajectoryPredictor

_SRL_CFG = OmegaConf.create({
    "collision_checker": {
        "vehicle": {"wheelbase": 2.7, "width": 1.8, "length": 4.5},
        "horizon_steps": 5, "step_size": 0.5,
        "action_lateral_offsets": {0: 0.0, 1: -0.5, 2: 0.5},
        "lane_half_width": 1.75,
    }
})


class TestTrajectoryPredictor:
    def test_horizon_length(self):
        pred = TrajectoryPredictor(_SRL_CFG)
        traj = pred.predict(0, (0.0, 0.0))
        assert len(traj) == 5

    def test_maintain_lane_zero_lateral(self):
        pred = TrajectoryPredictor(_SRL_CFG)
        traj = pred.predict(0, (0.0, 0.0))
        xs = [pt[0] for pt in traj]
        assert all(abs(x) < 0.01 for x in xs)

    def test_left_shift_negative_x(self):
        pred = TrajectoryPredictor(_SRL_CFG)
        traj = pred.predict(1, (0.0, 0.0))
        xs = [pt[0] for pt in traj]
        assert xs[-1] < 0

    def test_right_shift_positive_x(self):
        pred = TrajectoryPredictor(_SRL_CFG)
        traj = pred.predict(2, (0.0, 0.0))
        xs = [pt[0] for pt in traj]
        assert xs[-1] > 0

    def test_all_trajectories_returns_3(self):
        pred = TrajectoryPredictor(_SRL_CFG)
        all_t = pred.all_trajectories()
        assert set(all_t.keys()) == {0, 1, 2}


class TestCollisionChecker:
    def _empty_mask(self): return np.zeros((512, 512), dtype=np.uint8)
    def _full_mask(self):  return np.ones((512, 512), dtype=np.uint8)

    def test_no_collision_empty_mask(self):
        cc = CollisionChecker(_SRL_CFG)
        traj = [(0.0, 0.5), (0.0, 1.0), (0.0, 1.5), (0.0, 2.0), (0.0, 2.5)]
        collides, pts = cc.check(traj, self._empty_mask())
        assert collides is False
        assert len(pts) == 0

    def test_collision_full_mask(self):
        cc = CollisionChecker(_SRL_CFG)
        traj = [(0.0, 0.5), (0.0, 1.0)]
        collides, _ = cc.check(traj, self._full_mask())
        assert collides is True

    def test_all_actions_checked(self):
        cc = CollisionChecker(_SRL_CFG)
        result = cc.check_all_actions(self._empty_mask())
        assert set(result.keys()) == {0, 1, 2}
