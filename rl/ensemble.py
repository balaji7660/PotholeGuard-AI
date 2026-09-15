"""
rl/ensemble.py
Soft-Voting Ensemble over 4 RL agents.

P_ensemble(a|s) = Σ_i  w_i * P_i(a|s)
a_RL = argmax_a P_ensemble(a|s)

Initial equal weights: w_PPO = w_A2C = w_TRPO = w_RPPO = 0.25
Weights are configurable and can be replaced with performance-based adaptive weights.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Canonical agent name order
AGENT_NAMES = ["ppo", "a2c", "trpo", "recurrent_ppo"]


class SoftVotingEnsemble:
    """
    Soft-voting ensemble over multiple RL agents.

    Parameters
    ----------
    agents : dict[str, agent]
        Mapping from agent name to agent instance.
        Each agent must implement get_action_probs(state) → (3,) float32.
    cfg : DictConfig
        rl_config.yaml → ensemble section.
    """

    def __init__(self, agents: Optional[Dict[str, object]] = None, cfg: Optional[Any] = None) -> None:
        self.agents = agents or {}
        if cfg is not None and hasattr(cfg, "ensemble"):
            self.cfg = cfg.ensemble
            w_cfg = getattr(self.cfg, "weights", {})
            self._weights: Dict[str, float] = {
                name: float(w_cfg.get(name, 1.0 / max(1, len(self.agents))))
                for name in self.agents
            } if self.agents else {"ppo": 0.25, "a2c": 0.25, "trpo": 0.25, "recurrent_ppo": 0.25}
            self.adaptive = bool(getattr(self.cfg, "adaptive_weights", False))
        else:
            self.cfg = None
            self._weights = {
                name: 1.0 / max(1, len(self.agents)) for name in self.agents
            } if self.agents else {"ppo": 0.25, "a2c": 0.25, "trpo": 0.25, "recurrent_ppo": 0.25}
            self.adaptive = False

        self._normalise_weights()
        self._agent_rewards: Dict[str, List[float]] = {n: [] for n in self.agents}
        log.info(f"Ensemble weights: {self._weights}")

    # ------------------------------------------------------------------

    def _normalise_weights(self) -> None:
        total = sum(self._weights.values())
        self._weights = {k: v / total for k, v in self._weights.items()}

    # ------------------------------------------------------------------

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Compute weighted ensemble action probability distribution.

        Parameters
        ----------
        state : (117,) float32

        Returns
        -------
        ensemble_probs : (3,) float32 — probability over {Maintain, Left, Right}
        """
        ensemble = np.zeros(3, dtype=np.float32)

        for name, agent in self.agents.items():
            probs  = agent.get_action_probs(state)   # (3,)
            weight = self._weights.get(name, 0.0)
            ensemble += weight * probs

        # Normalise (should already sum to 1 if weights normalised)
        total = ensemble.sum()
        if total > 0:
            ensemble /= total

        return ensemble

    def select_action(self, state: np.ndarray) -> int:
        """
        Deterministically select action from ensemble.

        Returns
        -------
        action : int  ∈ {0, 1, 2}
        """
        probs = self.get_action_probs(state)
        return int(np.argmax(probs))

    # ------------------------------------------------------------------
    # Adaptive weight update
    # ------------------------------------------------------------------

    def record_reward(self, agent_name: str, reward: float) -> None:
        """
        Record per-agent reward for adaptive weight computation.
        Call after each step if adaptive_weights=True.
        """
        if agent_name in self._agent_rewards:
            self._agent_rewards[agent_name].append(reward)

    def update_adaptive_weights(self, window: int = 100) -> None:
        """
        Update weights proportional to recent mean reward per agent.
        Implementation choice — activated only when adaptive_weights=True in config.
        """
        if not self.adaptive:
            return

        means: Dict[str, float] = {}
        for name, rewards in self._agent_rewards.items():
            if len(rewards) >= window:
                means[name] = float(np.mean(rewards[-window:]))
            else:
                means[name] = float(np.mean(rewards)) if rewards else 1.0

        # Shift to positive
        min_val = min(means.values())
        means   = {k: v - min_val + 1e-6 for k, v in means.items()}
        total   = sum(means.values())
        self._weights = {k: v / total for k, v in means.items()}
        log.info(f"Adaptive ensemble weights updated: {self._weights}")

    # ------------------------------------------------------------------

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)
