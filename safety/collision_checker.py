"""
safety/collision_checker.py
Checks whether a predicted vehicle trajectory intersects with pothole regions.

The pothole mask (binary image) is mapped to a real-world coordinate grid.
Each trajectory waypoint is checked for intersection with the pothole footprint.

This module is intentionally modular — the vehicle model and coordinate mapping
can be updated independently for higher-fidelity simulations.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from omegaconf import DictConfig

from safety.trajectory import TrajectoryPredictor


class CollisionChecker:
    """
    Checks if a vehicle trajectory collides with pothole regions.

    Parameters
    ----------
    srl_cfg : DictConfig  (srl_config.yaml root)
    """

    def __init__(self, srl_cfg: DictConfig) -> None:
        self.traj_pred = TrajectoryPredictor(srl_cfg)
        cc_cfg         = srl_cfg.collision_checker
        self.veh_w     = float(cc_cfg.vehicle.width)
        self.veh_len   = float(cc_cfg.vehicle.length)
        self.lane_hw   = float(cc_cfg.lane_half_width)
        self.step_size = float(cc_cfg.step_size)

    def check(
        self,
        trajectory: List[Tuple[float, float]],
        pothole_mask: np.ndarray,
        image_size: Tuple[int, int] = (512, 512),
    ) -> Tuple[bool, List[Tuple[float, float]]]:
        """
        Check if trajectory collides with potholes.

        Parameters
        ----------
        trajectory    : list of (x_metres, y_metres) waypoints
        pothole_mask  : (H, W) binary uint8 — 1 = pothole
        image_size    : (H, W) of the mask image

        Returns
        -------
        (collision: bool, collision_points: list of (x,y) world coords)
        """
        H, W = image_size if image_size else pothole_mask.shape[:2]
        collision_points = []

        for x_m, y_m in trajectory:
            # Map world coords to image pixel coords
            # x : lateral [-lane_hw, +lane_hw] → [0, W]
            # y : forward [0, horizon*step]    → [H, 0]  (image top = far)
            px = int((x_m + self.lane_hw) / (2 * self.lane_hw) * W)
            py = int(H - min(y_m / (self.traj_pred.horizon * self.step_size), 1.0) * H)

            # Vehicle footprint: expand by half width
            half_w_px = max(1, int(self.veh_w / (2 * self.lane_hw) * W))

            px1 = max(0, px - half_w_px)
            px2 = min(W - 1, px + half_w_px)
            py1 = max(0, py - 5)
            py2 = min(H - 1, py + 5)

            if pothole_mask[py1:py2+1, px1:px2+1].any():
                collision_points.append((x_m, y_m))

        return (len(collision_points) > 0), collision_points

    def check_action(
        self,
        action: int,
        pothole_mask: np.ndarray,
        current_position: Tuple[float, float] = (0.0, 0.0),
        image_size: Tuple[int, int] = (512, 512),
    ) -> Tuple[bool, List[Tuple[float, float]]]:
        """
        Convenience: predict trajectory for *action* then check collision.
        """
        trajectory = self.traj_pred.predict(action, current_position)
        return self.check(trajectory, pothole_mask, image_size)

    def check_all_actions(
        self,
        pothole_mask: np.ndarray,
        current_position: Tuple[float, float] = (0.0, 0.0),
        image_size: Tuple[int, int] = (512, 512),
    ) -> dict:
        """
        Returns collision status for all three actions.

        Returns
        -------
        dict: {0: bool, 1: bool, 2: bool}  True = collision
        """
        results = {}
        for action in [0, 1, 2]:
            collision, _ = self.check_action(action, pothole_mask, current_position, image_size)
            results[action] = collision
        return results
