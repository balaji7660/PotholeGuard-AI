"""
state/temporal_smoothing.py
Exponential Moving Average (EMA) temporal smoother for UASA state features.

Reduces frame-to-frame noise in perception outputs before feeding into the
RL state vector.  Implementation choice — alpha loaded from config.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class EMATemporalSmoother:
    """
    Per-feature EMA smoother.

    EMA formula (implementation choice):
        x_smooth[t] = alpha * x_raw[t] + (1 - alpha) * x_smooth[t-1]

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector (117 for UASA).
    alpha : float
        Smoothing factor in (0, 1].
        alpha = 1.0  → no smoothing (use raw values)
        alpha → 0    → very heavy smoothing (slow to respond)
    """

    def __init__(self, state_dim: int, alpha: float = 0.7) -> None:
        self.state_dim = state_dim
        self.alpha     = float(alpha)
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset state (call at the start of each new episode)."""
        self._prev = None

    def __call__(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        raw_state : (state_dim,) float32

        Returns
        -------
        smoothed_state : (state_dim,) float32
        """
        if self._prev is None:
            self._prev = raw_state.copy()
            return raw_state.copy()

        smoothed = self.alpha * raw_state + (1.0 - self.alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed

    def update_alpha(self, new_alpha: float) -> None:
        """Dynamically update smoothing factor."""
        self.alpha = float(new_alpha)
