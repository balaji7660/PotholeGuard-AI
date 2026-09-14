"""
simulation/vehicle.py
Simplified vehicle state manager for the simulated driving scenario.
Maintains lateral position, speed, action history, and lateral deviation metrics.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np


ACTION_LATERAL = {0: 0.0, 1: -0.15, 2: 0.15}  # lateral shift per step (norm units)
ACTION_NAMES   = {0: "Maintain Lane", 1: "Shift Left", 2: "Shift Right"}


class SimulatedVehicle:
    """
    Simulated vehicle for the dataset-driven prototype.

    State:
      x_norm  : lateral position in [-1, 1]  (0 = lane centre)
      y_norm  : longitudinal progress [0, 1]
      speed   : forward speed (normalised, [0,1])
    """

    def __init__(
        self,
        start_x: float = 0.0,
        start_y: float = 0.7,
        speed: float = 0.5,
        lane_half_width: float = 1.75,
    ) -> None:
        self.x_norm = start_x
        self.y_norm = start_y
        self.speed  = speed
        self.lane_hw = lane_half_width

        self._action_history: List[int] = []
        self._x_history:      List[float] = [start_x]
        self._lateral_deviations: List[float] = []
        self.emergency_active: bool = False

    # ------------------------------------------------------------------

    def apply_action(self, action: int, emergency: bool = False) -> None:
        """
        Update vehicle position based on action.

        Parameters
        ----------
        action    : int ∈ {0, 1, 2}
        emergency : bool — True to apply emergency brake (stop movement)
        """
        self.emergency_active = emergency
        if emergency:
            self.speed = max(0.0, self.speed - 0.4)
            return

        # Lateral shift
        dx = ACTION_LATERAL.get(action, 0.0)
        self.x_norm = float(np.clip(self.x_norm + dx, -1.0, 1.0))

        # Forward progress
        self.y_norm = max(0.0, self.y_norm - self.speed * 0.05)  # move forward (up in view)

        self._action_history.append(action)
        self._x_history.append(self.x_norm)
        self._lateral_deviations.append(abs(self.x_norm) * self.lane_hw)

    def reset(
        self,
        start_x: float = 0.0,
        start_y: float = 0.7,
        speed: float = 0.5,
    ) -> None:
        self.x_norm = start_x
        self.y_norm = start_y
        self.speed  = speed
        self.emergency_active = False
        self._action_history = []
        self._x_history = [start_x]
        self._lateral_deviations = []

    @property
    def lateral_deviation_m(self) -> float:
        """Current lateral deviation from lane centre in metres."""
        return abs(self.x_norm) * self.lane_hw

    @property
    def mean_lateral_deviation(self) -> float:
        if not self._lateral_deviations:
            return 0.0
        return float(np.mean(self._lateral_deviations))

    @property
    def action_distribution(self) -> dict:
        """Fraction of time each action was selected."""
        n = len(self._action_history)
        if n == 0:
            return {0: 0.0, 1: 0.0, 2: 0.0}
        return {a: self._action_history.count(a) / n for a in [0, 1, 2]}

    @property
    def position(self) -> dict:
        return {"x_norm": self.x_norm, "y_norm": self.y_norm}
