"""
rl/ppo_agent.py
PPO agent wrapper around Stable-Baselines3.

The agent outputs a full action probability distribution (needed for soft voting).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

log = logging.getLogger(__name__)


class PPOAgent:
    """
    Wrapper for SB3 PPO.

    Parameters
    ----------
    env : gym.Env
        A PotholeAvoidanceEnv instance.
    cfg : DictConfig
        rl_config.yaml root.
    device : str
        "auto" | "cpu" | "cuda"
    """

    def __init__(self, env, cfg: DictConfig, device: str = "auto") -> None:
        self.cfg = cfg.ppo
        self.env = env
        self.device = device

        self.model = PPO(
            policy=self.cfg.policy,
            env=env,
            learning_rate=self.cfg.learning_rate,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_range=self.cfg.clip_range,
            ent_coef=self.cfg.ent_coef,
            vf_coef=self.cfg.vf_coef,
            max_grad_norm=self.cfg.max_grad_norm,
            device=device,
            verbose=1,
        )

    def train(self) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cb = CheckpointCallback(
            save_freq=10000,
            save_path=str(ckpt_dir),
            name_prefix="ppo_ckpt",
        )
        log.info(f"Training PPO for {self.cfg.total_timesteps} timesteps.")
        self.model.learn(
            total_timesteps=self.cfg.total_timesteps,
            callback=cb,
        )

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Return softmax action probabilities for the given state.

        Parameters
        ----------
        state : (117,) float32

        Returns
        -------
        probs : (3,) float32
        """
        obs_tensor = torch.as_tensor(state[np.newaxis], device=self.model.device)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.squeeze().cpu().numpy()
        return probs.astype(np.float32)

    def predict(self, state: np.ndarray) -> int:
        """Greedy action selection."""
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)
        log.info(f"PPO saved to {path}")

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self.env, device=self.device)
        log.info(f"PPO loaded from {path}")
