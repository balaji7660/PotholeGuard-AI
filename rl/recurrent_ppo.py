"""
rl/recurrent_ppo.py
Recurrent PPO (LSTM) agent using sb3-contrib RecurrentPPO.

Uses MlpLstmPolicy for temporal reasoning across frames.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

log = logging.getLogger(__name__)


class RecurrentPPOAgent:
    """
    Wrapper for sb3-contrib RecurrentPPO (LSTM memory).

    Parameters
    ----------
    env : gym.Env
    cfg : DictConfig  (rl_config.yaml)
    device : str
    """

    def __init__(self, env, cfg: DictConfig, device: str = "auto") -> None:
        self.cfg    = cfg.recurrent_ppo
        self.env    = env
        self.device = device

        policy_kwargs = {
            "lstm_hidden_size": self.cfg.lstm_hidden_size,
            "n_lstm_layers":    self.cfg.n_lstm_layers,
        }

        self.model = RecurrentPPO(
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
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
        )

        # Persistent LSTM state for inference
        self._lstm_states = None
        self._episode_starts = np.ones((1,), dtype=bool)

    def train(self) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cb = CheckpointCallback(
            save_freq=10000,
            save_path=str(ckpt_dir),
            name_prefix="rppo_ckpt",
        )
        log.info(f"Training RecurrentPPO for {self.cfg.total_timesteps} timesteps.")
        self.model.learn(
            total_timesteps=self.cfg.total_timesteps,
            callback=cb,
        )

    def reset_lstm(self) -> None:
        """Reset LSTM state at the start of a new episode."""
        self._lstm_states  = None
        self._episode_starts = np.ones((1,), dtype=bool)

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Return action probabilities using the LSTM policy.
        Maintains LSTM state across calls within an episode.
        """
        obs   = state[np.newaxis]  # (1, 117)
        action, self._lstm_states = self.model.predict(
            obs,
            state=self._lstm_states,
            episode_start=self._episode_starts,
            deterministic=False,
        )
        self._episode_starts = np.zeros((1,), dtype=bool)

        # Retrieve distribution to get probabilities
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device)
        lstm_states_tensor = self._lstm_states
        with torch.no_grad():
            features, _ = self.model.policy.extract_features(obs_tensor)
            # Fallback: use uniform distribution if lstm probs not extractable cleanly
            try:
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.squeeze().cpu().numpy()
            except Exception:
                # Fallback to one-hot encoding for the predicted action
                probs = np.zeros(3, dtype=np.float32)
                probs[int(action[0])] = 1.0

        return probs.astype(np.float32)

    def predict(self, state: np.ndarray) -> int:
        action, self._lstm_states = self.model.predict(
            state[np.newaxis],
            state=self._lstm_states,
            episode_start=self._episode_starts,
            deterministic=True,
        )
        self._episode_starts = np.zeros((1,), dtype=bool)
        return int(action[0])

    def save(self, path: str) -> None:
        self.model.save(path)
        log.info(f"RecurrentPPO saved to {path}")

    def load(self, path: str) -> None:
        self.model = RecurrentPPO.load(path, env=self.env, device=self.device)
        log.info(f"RecurrentPPO loaded from {path}")
