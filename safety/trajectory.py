"""
safety/trajectory.py
Short-horizon vehicle trajectory predictor.

For each candidate action, computes the expected sequence of (x, y) positions
over a short look-ahead horizon using a simplified kinematic model.

This is an IMPLEMENTATION CHOICE — a realistic vehicle dynamics model
(bicycle model, MPC, etc.) can replace this module without changing the SRL interface.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from omegaconf import DictConfig

# Action → lateral steering increment (implementation choice, tunable in config)
_DEFAULT_LATERAL_OFFSETS = {0: 0.0, 1: -0.5, 2: 0.5}  # metres


class TrajectoryPredictor:
    """
    Predicts short-horizon (x, y) vehicle trajectory for a given action.

    Coordinate system:
      x : lateral position (positive = right of lane centre)
      y : longitudinal position (positive = forward)

    Parameters
    ----------
    srl_cfg : DictConfig  (srl_config.yaml root)
    """

    def __init__(self, srl_cfg: DictConfig) -> None:
        cc_cfg = srl_cfg.collision_checker
        self.horizon   : int   = int(cc_cfg.horizon_steps)
        self.step_size : float = float(cc_cfg.step_size)
        self.lat_offsets: dict = dict(cc_cfg.action_lateral_offsets)
        self.lane_hw    : float = float(cc_cfg.lane_half_width)
        self.veh_w      : float = float(cc_cfg.vehicle.width)

    def predict(
        self,
        action: int,
        current_position: Tuple[float, float] = (0.0, 0.0),
        current_speed: float = 1.0,
    ) -> List[Tuple[float, float]]:
        """
        Generate trajectory waypoints for the given action.

        Parameters
        ----------
        action           : int  {0, 1, 2}
        current_position : (x, y) in metres
        current_speed    : forward speed (metres per step)

        Returns
        -------
        waypoints : list of (x, y) tuples, length = horizon_steps
        """
        x0, y0 = current_position
        lat_target = float(self.lat_offsets.get(action, 0.0))

        waypoints = []
        x, y = x0, y0

        for step in range(self.horizon):
            # Smooth lateral transition towards target offset
            t = (step + 1) / self.horizon
            x_new = x0 + lat_target * t
            y_new = y0 + current_speed * self.step_size * (step + 1)
            waypoints.append((float(x_new), float(y_new)))

        return waypoints

    def all_trajectories(
        self,
        current_position: Tuple[float, float] = (0.0, 0.0),
    ) -> dict:
        """Return trajectories for all three actions."""
        return {
            a: self.predict(a, current_position)
            for a in [0, 1, 2]
        }

    def is_within_lane(self, trajectory: List[Tuple[float, float]]) -> bool:
        """Check if all waypoints stay within lane boundaries."""
        for x, _ in trajectory:
            if abs(x) > (self.lane_hw - self.veh_w / 2.0):
                return False
        return True
