"""
safety/srl.py
Safety Refinement Layer (SRL) — deterministic, RL-independent safety gate.

The SRL receives:
  - Proposed RL action
  - Perception outputs (uncertainty, severity)
  - Vehicle state (lateral deviation)
  - Pothole mask (for collision checking)

It verifies the action against 7 safety criteria and either:
  (a) Accepts the RL action as-is, or
  (b) Overrides it with a deterministic safe fallback.

All thresholds are loaded from configs/srl_config.yaml.
The SRL is COMPLETELY INDEPENDENT of the RL policy.

Override fallback priority:
  1. Emergency brake  (pothole too close / extreme severity)
  2. Centre lane      (collision on all lateral actions)
  3. Conservative action (uncertainty too high)
  4. Configured per-criterion fallback
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from safety.collision_checker import CollisionChecker
from safety.trajectory import TrajectoryPredictor

# Action labels
ACTION_LABELS = {0: "Maintain Lane", 1: "Shift Left", 2: "Shift Right"}
FALLBACK_LABELS = {
    "emergency_brake": "Emergency Brake",
    "centre_lane":     "Maintain / Centre Lane",
    "slow_down":       "Slow Down",
    "conservative":    "Conservative Action",
}


@dataclass
class SRLDecision:
    """Full SRL output for one inference step."""
    rl_action:        int
    final_action:     int
    accepted:         bool
    override_reason:  Optional[str]            = None
    failed_checks:    List[str]                = field(default_factory=list)
    collision_map:    dict                     = field(default_factory=dict)  # {action: bool}
    uncertainty_ok:   bool                     = True
    severity_ok:      bool                     = True
    deviation_ok:     bool                     = True
    lane_ok:          bool                     = True
    consistency_ok:   bool                     = True

    @property
    def label(self) -> str:
        return ACTION_LABELS.get(self.final_action, str(self.final_action))

    @property
    def rl_label(self) -> str:
        return ACTION_LABELS.get(self.rl_action, str(self.rl_action))


class SafetyRefinementLayer:
    """
    Deterministic safety gate applied after the ensemble RL decision.

    Parameters
    ----------
    srl_cfg : DictConfig  (srl_config.yaml root)
    """

    def __init__(self, srl_cfg: DictConfig) -> None:
        cfg = srl_cfg.srl
        self.enabled            = bool(cfg.enabled)
        self.unc_threshold      = float(cfg.uncertainty_threshold)
        self.sev_threshold      = float(cfg.severity_threshold)
        self.lat_dev_threshold  = float(cfg.lateral_deviation_threshold)
        self.collision_threshold= float(cfg.collision_risk_threshold)
        self.consistency_window = int(cfg.action_consistency_window)
        self.osc_threshold      = int(cfg.oscillation_count_threshold)

        fb = cfg.fallback
        self.fb_uncertainty  = int(fb.uncertainty_override)
        self.fb_severity     = int(fb.severity_override)
        self.fb_collision    = int(fb.collision_override)
        self.fb_oscillation  = int(fb.oscillation_override)

        em = cfg.emergency_brake
        self.emergency_enabled   = bool(em.enabled)
        self.emergency_min_depth = float(em.min_depth_threshold)
        self.emergency_sev_thr   = float(em.severity_threshold)

        self.collision_checker = CollisionChecker(srl_cfg)
        self.traj_pred         = TrajectoryPredictor(srl_cfg)

        # Action history for oscillation detection
        self._action_history: List[int] = []

        # SRL intervention counter
        self.total_calls:      int = 0
        self.total_overrides:  int = 0

    @property
    def intervention_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_overrides / self.total_calls

    def reset(self) -> None:
        self._action_history = []
        self.total_calls = 0
        self.total_overrides = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        rl_action:         int,
        uncertainty_score: float,
        severity_score:    float,
        lateral_deviation: float,
        min_depth:         float,
        pothole_mask:      np.ndarray,
        current_position:  Tuple[float, float] = (0.0, 0.0),
    ) -> SRLDecision:
        """
        Evaluate the RL-proposed action against all safety criteria.

        Parameters
        ----------
        rl_action         : int ∈ {0,1,2}
        uncertainty_score : float ∈ [0,1] — perception uncertainty
        severity_score    : float ∈ [0,1] — max pothole severity
        lateral_deviation : float (metres) — current lateral offset from centre
        min_depth         : float (metres) — minimum depth in scene
        pothole_mask      : (H,W) uint8 binary mask
        current_position  : (x_m, y_m)

        Returns
        -------
        SRLDecision
        """
        self.total_calls += 1
        failed_checks: List[str] = []

        if not self.enabled:
            return SRLDecision(rl_action=rl_action, final_action=rl_action, accepted=True)

        # Check 1: Perception uncertainty
        unc_ok = uncertainty_score <= self.unc_threshold

        # Check 2: Pothole severity
        sev_ok = severity_score <= self.sev_threshold

        # Check 3: Lateral deviation
        dev_ok = abs(lateral_deviation) <= self.lat_dev_threshold

        # Check 4: Lane boundary (trajectory stays within lane)
        traj = self.traj_pred.predict(rl_action, current_position)
        lane_ok = self.traj_pred.is_within_lane(traj)

        # Check 5 & 6: Collision risk for proposed action
        H, W = pothole_mask.shape[:2]
        collision_map = self.collision_checker.check_all_actions(
            pothole_mask, current_position, (H, W)
        )
        proposed_collides = collision_map.get(rl_action, False)

        # Check 7: Action consistency (oscillation detection)
        consistency_ok = self._check_consistency(rl_action)

        # Emergency brake: extreme severity + very close depth
        emergency = (
            self.emergency_enabled
            and severity_score >= self.emergency_sev_thr
            and min_depth <= self.emergency_min_depth
        )

        # Collect failed checks
        if not unc_ok:      failed_checks.append("High perception uncertainty")
        if not sev_ok:      failed_checks.append("Extreme pothole severity")
        if not dev_ok:      failed_checks.append("Excessive lateral deviation")
        if not lane_ok:     failed_checks.append("Trajectory exits lane boundary")
        if proposed_collides: failed_checks.append("Collision risk on proposed trajectory")
        if not consistency_ok: failed_checks.append("Action oscillation detected")
        if emergency:        failed_checks.append("EMERGENCY: obstacle too close")

        is_safe = len(failed_checks) == 0

        if is_safe:
            self._action_history.append(rl_action)
            return SRLDecision(
                rl_action=rl_action,
                final_action=rl_action,
                accepted=True,
                collision_map=collision_map,
                uncertainty_ok=unc_ok,
                severity_ok=sev_ok,
                deviation_ok=dev_ok,
                lane_ok=lane_ok,
                consistency_ok=consistency_ok,
            )

        # --- Override selection ---
        self.total_overrides += 1

        if emergency:
            # Emergency brake: action 3 (virtual — mapped by vehicle sim)
            final_action = 0   # Maintain lane + signal slow down separately
            reason = "EMERGENCY BRAKE — obstacle too close"
        elif proposed_collides:
            # Try to find a safe lateral action
            final_action = self._find_safe_action(collision_map, rl_action)
            reason = "Collision risk on proposed trajectory"
        elif not unc_ok:
            final_action = self.fb_uncertainty
            reason = "High perception uncertainty → conservative action"
        elif not sev_ok:
            final_action = self.fb_severity
            reason = "Extreme pothole severity"
        elif not lane_ok:
            final_action = 0  # Maintain lane
            reason = "Trajectory exits lane boundary"
        elif not consistency_ok:
            final_action = self.fb_oscillation
            reason = "Action oscillation detected"
        else:
            final_action = 0
            reason = "General safety override"

        self._action_history.append(final_action)
        return SRLDecision(
            rl_action=rl_action,
            final_action=final_action,
            accepted=False,
            override_reason=reason,
            failed_checks=failed_checks,
            collision_map=collision_map,
            uncertainty_ok=unc_ok,
            severity_ok=sev_ok,
            deviation_ok=dev_ok,
            lane_ok=lane_ok,
            consistency_ok=consistency_ok,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_safe_action(self, collision_map: dict, preferred: int) -> int:
        """
        Select the safest available action given collision map.
        Priority: preferred → opposite lateral → maintain.
        """
        for action in [preferred, 2 - preferred, 0]:  # try preferred, opposite, maintain
            if not collision_map.get(action, False):
                return action
        return 0  # all collide — maintain lane (least harm)

    def _check_consistency(self, action: int) -> bool:
        """Detect oscillation: action flips more than threshold in recent window."""
        h = self._action_history[-self.consistency_window:]
        if len(h) < 2:
            return True
        flips = sum(1 for i in range(1, len(h)) if h[i] != h[i-1])
        return flips < self.osc_threshold

    # ------------------------------------------------------------------
    # Scenario presets (for GUI demonstration)
    # ------------------------------------------------------------------

    def scenario_a(self, pothole_mask: np.ndarray) -> SRLDecision:
        """Scenario A: RL picks Shift Left, safe — SRL accepts."""
        mask_no_left = np.zeros_like(pothole_mask)  # no pothole on left
        mask_no_left[:, pothole_mask.shape[1]//2:] = pothole_mask[:, pothole_mask.shape[1]//2:]
        return self.evaluate(1, 0.2, 0.3, 0.1, 5.0, mask_no_left)

    def scenario_b(self, pothole_mask: np.ndarray) -> SRLDecision:
        """Scenario B: RL picks Shift Left, left has pothole — SRL overrides to Right."""
        mask_left = pothole_mask.copy()
        return self.evaluate(1, 0.3, 0.5, 0.1, 3.0, mask_left)

    def scenario_c(self, pothole_mask: np.ndarray) -> SRLDecision:
        """Scenario C: High uncertainty — SRL overrides aggressive maneuver."""
        return self.evaluate(1, 0.85, 0.4, 0.2, 5.0, pothole_mask)

    def scenario_d(self, pothole_mask: np.ndarray) -> SRLDecision:
        """Scenario D: Pothole very close — SRL triggers Emergency Brake."""
        return self.evaluate(0, 0.4, 0.97, 0.0, 0.3, pothole_mask)
