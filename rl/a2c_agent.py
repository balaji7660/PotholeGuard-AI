"""
rl/a2c_agent.py
A2C agent wrapper around Stable-Baselines3.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback

log = logging.getLogger(__name__)


class A2CAgent:
    """
    Wrapper for SB3 A2C.

    Parameters
    ----------
    env : gym.Env
    cfg : DictConfig  (rl_config.yaml)
    device : str
    """

    def __init__(self, env, cfg: DictConfig, device: str = "auto") -> None:
        self.cfg    = cfg.a2c
        self.env    = env
        self.device = device

        self.model = A2C(
            policy=self.cfg.policy,
            env=env,
            learning_rate=self.cfg.learning_rate,
            n_steps=self.cfg.n_steps,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            ent_coef=self.cfg.ent_coef,
            vf_coef=self.cfg.vf_coef,
            max_grad_norm=self.cfg.max_grad_norm,
            rms_prop_eps=self.cfg.rms_prop_eps,
            device=device,
            verbose=1,
        )

    def train(self) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cb = CheckpointCallback(
            save_freq=10000,
            save_path=str(ckpt_dir),
            name_prefix="a2c_ckpt",
        )
        log.info(f"Training A2C for {self.cfg.total_timesteps} timesteps.")
        self.model.learn(
            total_timesteps=self.cfg.total_timesteps,
            callback=cb,
        )

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Return softmax action probabilities."""
        obs_tensor = torch.as_tensor(state[np.newaxis], device=self.model.device)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.squeeze().cpu().numpy()
        return probs.astype(np.float32)

    def predict(self, state: np.ndarray) -> int:
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)
        log.info(f"A2C saved to {path}")

    def load(self, path: str) -> None:
        self.model = A2C.load(path, env=self.env, device=self.device)
        log.info(f"A2C loaded from {path}")
