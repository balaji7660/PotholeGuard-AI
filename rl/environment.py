"""
rl/environment.py
Modular Gymnasium environment for pothole avoidance.

The environment is designed to be simulator-agnostic:
  - In stub mode: uses pre-recorded perception outputs or random synthetic data.
  - CARLA mode: swap _get_observation() to query CARLA server.
  - Custom sim: override PotholeAvoidanceEnv and implement _step_simulator().

Action Space:
  Discrete(3)  →  0=Maintain Lane, 1=Shift Left, 2=Shift Right

Observation Space:
  Box(117,)  →  UASA 117-dimensional state vector

State transitions and collision are simulated here via the collision checker.
In a full deployment, this environment class is replaced by a CARLA bridge.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from rl.reward import RewardCalculator
from safety.collision_checker import CollisionChecker
from safety.trajectory import TrajectoryPredictor
from state.uasa import UASAStateGenerator, STATE_DIM


class PotholeAvoidanceEnv(gym.Env):
    """
    Pothole Avoidance gymnasium environment.

    Parameters
    ----------
    rl_cfg        : DictConfig  (rl_config.yaml)
    severity_cfg  : DictConfig  (severity_config.yaml)
    srl_cfg       : DictConfig  (srl_config.yaml)
    perception_pipeline : optional PerceptionPipeline
        If provided, real images from a dataset will be used as observations.
        If None, random synthetic states are generated (stub/unit-test mode).
    image_paths : list[str] | None
        Ordered list of image file paths to step through.
    seed : int | None
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        rl_cfg:       DictConfig,
        severity_cfg: DictConfig,
        srl_cfg:      DictConfig,
        perception_pipeline=None,  # PerceptionPipeline (avoid circular import)
        image_paths: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.rl_cfg   = rl_cfg
        self.env_cfg  = rl_cfg.environment
        self._rng     = np.random.default_rng(seed or self.env_cfg.seed)

        # Spaces
        self.action_space = spaces.Discrete(self.env_cfg.action_dim)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )

        # Modules
        self.reward_calc = RewardCalculator(rl_cfg)
        self.uasa_gen    = UASAStateGenerator(
            severity_cfg=severity_cfg,
            rl_cfg=rl_cfg,
        )
        self.collision_checker = CollisionChecker(srl_cfg)
        self.trajectory_pred   = TrajectoryPredictor(srl_cfg)

        # Perception pipeline (optional)
        self.pipeline    = perception_pipeline
        self.image_paths = image_paths or []
        self._img_idx    = 0

        # Episode state
        self._step_count:  int = 0
        self._prev_action: int = 0
        self._current_state: Optional[np.ndarray] = None
        self._current_perception: dict = {}

        # Action smoothing
        sm_cfg = self.env_cfg.action_smoothing
        self._ema_alpha   = float(sm_cfg.ema_alpha)
        self._min_persist = int(sm_cfg.min_persistence)
        self._action_history: list[int] = []

    # ------------------------------------------------------------------
    # Gymnasium protocol
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count  = 0
        self._prev_action = 0
        self._img_idx     = 0
        self._action_history = []
        self.uasa_gen.reset()

        obs = self._get_observation()
        self._current_state = obs
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Returns
        -------
        observation : (117,) state
        reward      : float
        terminated  : bool
        truncated   : bool
        info        : dict
        """
        # Apply action smoothing
        action = self._smooth_action(int(action))

        # Simulate environment step (advance image index or CARLA)
        self._advance_simulator()
        obs = self._get_observation()

        # Extract scalars from perception for reward calculation
        seg = self._current_perception.get("segmentation", np.zeros((512, 512)))
        unc = self._current_perception.get("uncertainty", np.zeros((512, 512)))
        dep = self._current_perception.get("depth", np.ones((512, 512)) * 10.0)

        max_severity   = float(obs[6])   # global feature: severity_max
        mean_uncertainty = float(obs[0]) # global feature: frame_uncertainty_mean

        # Collision check
        trajectory = self.trajectory_pred.predict(action, current_position=(0.0, 0.0))
        pothole_mask = (seg >= 0.5).astype(np.uint8)
        collision, _ = self.collision_checker.check(trajectory, pothole_mask)

        # Reward
        reward_info = self.reward_calc.compute(
            severity_score=max_severity,
            uncertainty_score=mean_uncertainty,
            current_action=action,
            prev_action=self._prev_action,
            collision=collision,
        )

        self._prev_action = action
        self._step_count += 1
        self._current_state = obs

        terminated = collision
        truncated  = self._step_count >= self.env_cfg.max_steps

        info = {
            "reward_breakdown": {
                "progress":    reward_info.progress,
                "severity":    reward_info.severity,
                "uncertainty": reward_info.uncertainty,
                "steering":    reward_info.steering,
                "collision":   reward_info.collision,
            },
            "collision": collision,
            "step": self._step_count,
            "action_taken": action,
        }

        return obs, float(reward_info.total), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return the current perception segmentation as RGB (for logging)."""
        seg = self._current_perception.get("segmentation", None)
        if seg is None:
            return None
        vis = (seg * 255).astype(np.uint8)
        import cv2
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
        return vis_rgb

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation generation
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Get current UASA state vector."""
        if self.pipeline is not None and self.image_paths:
            # Real perception from dataset images
            import cv2
            path = self.image_paths[self._img_idx % len(self.image_paths)]
            image_bgr = cv2.imread(str(path))
            if image_bgr is not None:
                self._current_perception = self.pipeline.infer(image_bgr)
            else:
                self._current_perception = self._synthetic_perception()
        else:
            # Stub mode: synthetic random perception
            self._current_perception = self._synthetic_perception()

        seg = self._current_perception["segmentation"]
        dep = self._current_perception["depth"]
        unc = self._current_perception["uncertainty"]

        state = self.uasa_gen.compute(seg, dep, unc, last_action=self._prev_action)
        return state

    def _synthetic_perception(self) -> dict:
        """Generate a random (but plausible) synthetic perception output for stub mode."""
        H, W = 512, 512
        seg = np.zeros((H, W), dtype=np.float32)
        dep = self._rng.uniform(5.0, 40.0, (H, W)).astype(np.float32)
        unc = self._rng.uniform(0.0, 0.3, (H, W)).astype(np.float32)

        # Add 1-3 random pothole blobs
        n_potholes = int(self._rng.integers(0, 4))
        for _ in range(n_potholes):
            cx = int(self._rng.integers(100, W - 100))
            cy = int(self._rng.integers(200, H - 50))
            rw = int(self._rng.integers(20, 80))
            rh = int(self._rng.integers(15, 50))
            import cv2
            cv2.ellipse(seg, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)
            dep_hole = self._rng.uniform(0.1, 2.0, (H, W)).astype(np.float32)
            mask = seg > 0.5
            dep[mask] = dep_hole[mask]
            unc[mask] = self._rng.uniform(0.4, 0.8)

        return {"segmentation": seg, "depth": dep, "uncertainty": unc}

    # ------------------------------------------------------------------
    # Simulator advance
    # ------------------------------------------------------------------

    def _advance_simulator(self) -> None:
        """Move to next image/frame."""
        self._img_idx += 1

    # ------------------------------------------------------------------
    # Action smoothing
    # ------------------------------------------------------------------

    def _smooth_action(self, action: int) -> int:
        """
        Enforce minimum action persistence to prevent rapid oscillation.
        Implementation choice — min_persistence loaded from config.
        """
        self._action_history.append(action)
        if not self.env_cfg.action_smoothing.enabled:
            return action

        # If the agent has been switching back-and-forth, enforce persistence
        if len(self._action_history) >= self._min_persist:
            recent = self._action_history[-self._min_persist:]
            if recent[-1] != recent[0] and self._prev_action != action:
                # Flip detected within persistence window — hold previous
                return self._prev_action

        return action
