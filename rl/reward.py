"""
rl/reward.py
Reward function for pothole avoidance.

R_total = R_progress
        - P_severity     * severity_score
        - P_uncertainty  * uncertainty_score
        - P_steering     * action_change_penalty
        - P_collision    (binary collision event)

NOTE: All coefficients are IMPLEMENTATION CHOICES loaded from config.
The research paper describes the structure of the reward function but does
NOT specify exact numerical coefficients. Every coefficient must be tuned
experimentally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig


@dataclass
class RewardInfo:
    """Detailed reward breakdown for logging/debugging."""
    total:       float
    progress:    float
    severity:    float
    uncertainty: float
    steering:    float
    collision:   float


class RewardCalculator:
    """
    Computes the per-step reward for the RL agent.

    Parameters
    ----------
    reward_cfg : DictConfig
        rl_config.yaml → reward section.
    """

    def __init__(self, reward_cfg: DictConfig) -> None:
        cfg = reward_cfg.reward
        self.r_progress   = float(cfg.progress_reward)
        self.p_severity   = float(cfg.severity_penalty)
        self.p_uncertainty= float(cfg.uncertainty_penalty)
        self.p_steering   = float(cfg.steering_penalty)
        self.p_collision  = float(cfg.collision_penalty)

        self.high_sev_thr = float(cfg.high_severity_threshold)
        self.high_unc_thr = float(cfg.high_uncertainty_threshold)

    # ------------------------------------------------------------------

    def compute(
        self,
        severity_score:    float,
        uncertainty_score: float,
        current_action:    int,
        prev_action:       int,
        collision:         bool,
        step_survived:     bool = True,
    ) -> RewardInfo:
        """
        Compute step reward.

        Parameters
        ----------
        severity_score    : float ∈ [0,1]  max pothole severity this step
        uncertainty_score : float ∈ [0,1]  mean perception uncertainty
        current_action    : int ∈ {0,1,2}  current RL action
        prev_action       : int ∈ {0,1,2}  previous RL action
        collision         : bool           True if vehicle collided with pothole
        step_survived     : bool           True if episode not terminated by env

        Returns
        -------
        RewardInfo
        """
        # Forward progress reward
        r_prog = self.r_progress if step_survived else 0.0

        # Severity penalty (proportional above threshold)
        sev_above = max(0.0, severity_score - self.high_sev_thr)
        p_sev = self.p_severity * sev_above

        # Uncertainty penalty (proportional above threshold)
        unc_above = max(0.0, uncertainty_score - self.high_unc_thr)
        p_unc = self.p_uncertainty * unc_above

        # Steering / abrupt action change penalty
        p_steer = self.p_steering if (current_action != prev_action) else 0.0

        # Collision penalty
        p_coll = self.p_collision if collision else 0.0

        total = r_prog - p_sev - p_unc - p_steer - p_coll

        return RewardInfo(
            total=total,
            progress=r_prog,
            severity=-p_sev,
            uncertainty=-p_unc,
            steering=-p_steer,
            collision=-p_coll,
        )
