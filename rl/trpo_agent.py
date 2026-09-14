"""
rl/trpo_agent.py
Custom TRPO (Trust Region Policy Optimisation) agent.

SB3 does not include TRPO, so this is a custom PyTorch implementation using:
  - GAE advantage estimation
  - Conjugate gradient for the natural gradient direction
  - Backtracking line search to satisfy the KL constraint

All hyperparameters are loaded from configs/rl_config.yaml → trpo section.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributions import Categorical

from state.uasa import STATE_DIM

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Actor-Critic network for TRPO
# ---------------------------------------------------------------------------

class TRPOActorCritic(nn.Module):
    """Shared trunk + separate actor/critic heads."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = 3) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.trunk(x)
        logits = self.actor(z)
        value  = self.critic(z).squeeze(-1)
        return logits, value

    def get_dist(self, x: torch.Tensor) -> Categorical:
        logits, _ = self(x)
        return Categorical(logits=logits)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self(x)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Simple rollout buffer for TRPO."""

    def __init__(self) -> None:
        self.states:  List[np.ndarray] = []
        self.actions: List[int]         = []
        self.rewards: List[float]       = []
        self.dones:   List[bool]        = []
        self.values:  List[float]       = []
        self.log_probs: List[float]     = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear(self) -> None:
        self.__init__()

    def compute_returns(self, gamma: float, gae_lambda: float) -> np.ndarray:
        """Compute GAE advantages and returns."""
        rewards  = np.array(self.rewards, dtype=np.float32)
        values   = np.array(self.values + [0.0], dtype=np.float32)
        dones    = np.array(self.dones, dtype=np.float32)
        n        = len(rewards)
        adv      = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            delta   = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            adv[t]  = last_gae

        returns = adv + values[:n]
        return adv, returns


# ---------------------------------------------------------------------------
# TRPO Agent
# ---------------------------------------------------------------------------

class TRPOAgent:
    """
    TRPO agent with conjugate gradient + backtracking line search.

    Parameters
    ----------
    env : gym.Env
    cfg : DictConfig  (rl_config.yaml)
    device : str
    """

    def __init__(self, env, cfg: DictConfig, device: str = "auto") -> None:
        self.cfg    = cfg.trpo
        self.env    = env
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else
            ("cpu" if device == "auto" else device)
        )

        action_dim = env.action_space.n
        self.model  = TRPOActorCritic(STATE_DIM, action_dim).to(self.device)
        self.value_optim = torch.optim.Adam(
            self.model.critic.parameters(), lr=self.cfg.learning_rate
        )
        self.buffer = RolloutBuffer()

        self.ckpt_dir = Path(self.cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        log.info(f"Training TRPO for {self.cfg.total_timesteps} timesteps.")
        total_steps = 0
        state, _ = self.env.reset()

        while total_steps < self.cfg.total_timesteps:
            # Collect batch_size steps
            for _ in range(self.cfg.batch_size):
                action, log_prob, value = self._select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.add(state, action, reward, done, value, log_prob)
                state = next_state if not done else self.env.reset()[0]
                total_steps += 1

            # Update policy
            self._update()
            self.buffer.clear()

            if total_steps % 50000 == 0:
                log.info(f"  TRPO step {total_steps}/{self.cfg.total_timesteps}")
                self.save(str(self.ckpt_dir / f"trpo_{total_steps}.pt"))

    def _select_action(
        self, state: np.ndarray
    ) -> Tuple[int, float, float]:
        obs   = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist  = self.model.get_dist(obs)
            action = dist.sample()
            lp     = dist.log_prob(action).item()
            _, val = self.model(obs)
            v      = val.item()
        return int(action.item()), lp, v

    def _update(self) -> None:
        """Single TRPO update step."""
        adv, returns = self.buffer.compute_returns(self.cfg.gamma, self.cfg.gae_lambda)

        states   = torch.tensor(np.array(self.buffer.states),  dtype=torch.float32, device=self.device)
        actions  = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        returns_t= torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t    = torch.tensor(adv,     dtype=torch.float32, device=self.device)
        adv_t    = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Value network update (standard gradient descent)
        _, values = self.model(states)
        val_loss  = F.mse_loss(values, returns_t)
        self.value_optim.zero_grad()
        val_loss.backward()
        self.value_optim.step()

        # Policy update via TRPO
        old_probs = self.model.get_action_probs(states).detach()
        old_log_probs = torch.log(old_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)

        def policy_loss() -> torch.Tensor:
            probs     = self.model.get_action_probs(states)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
            ratio     = torch.exp(log_probs - old_log_probs.detach())
            return -(ratio * adv_t).mean()

        def kl_divergence() -> torch.Tensor:
            probs     = self.model.get_action_probs(states)
            old_p     = old_probs.clamp(1e-8, 1.0)
            new_p     = probs.clamp(1e-8, 1.0)
            kl        = (old_p * (torch.log(old_p) - torch.log(new_p))).sum(-1)
            return kl.mean()

        # Compute natural gradient direction via conjugate gradient
        g   = self._flat_grad(policy_loss())
        direction = self._conjugate_gradient(kl_divergence, g)
        shs = 0.5 * (direction * self._hessian_vector_product(kl_divergence, direction)).sum()
        step_size = torch.sqrt(torch.tensor(self.cfg.max_kl, device=self.device) / (shs + 1e-8))
        full_step  = step_size * direction

        # Backtracking line search
        params_flat = self._get_flat_params()
        self._backtrack_step(params_flat, full_step, policy_loss, kl_divergence)

    # ------------------------------------------------------------------
    # TRPO helpers
    # ------------------------------------------------------------------

    def _flat_grad(self, loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(loss, self.model.actor.parameters(), create_graph=True)
        return torch.cat([g.contiguous().view(-1) for g in grads])

    def _hessian_vector_product(self, kl_fn, v: torch.Tensor) -> torch.Tensor:
        kl   = kl_fn()
        grads = torch.autograd.grad(kl, self.model.actor.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
        grad_v    = (flat_grad * v.detach()).sum()
        grads2    = torch.autograd.grad(grad_v, self.model.actor.parameters())
        flat_grad2= torch.cat([g.contiguous().view(-1) for g in grads2])
        return flat_grad2 + self.cfg.damping * v.detach()

    def _conjugate_gradient(self, kl_fn, b: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone().detach()
        p = r.clone()
        rdotr = r @ r
        for _ in range(self.cfg.cg_iters):
            Ap    = self._hessian_vector_product(kl_fn, p)
            alpha = rdotr / (p @ Ap + 1e-8)
            x     = x + alpha * p
            r     = r - alpha * Ap
            new_rdotr = r @ r
            beta  = new_rdotr / (rdotr + 1e-8)
            p     = r + beta * p
            rdotr = new_rdotr
        return x

    def _get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.contiguous().view(-1) for p in self.model.actor.parameters()])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self.model.actor.parameters():
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].view_as(p))
            offset += n

    def _backtrack_step(
        self,
        params_flat: torch.Tensor,
        full_step:   torch.Tensor,
        loss_fn,
        kl_fn,
    ) -> None:
        old_loss = loss_fn().item()
        coeff = 1.0
        for _ in range(self.cfg.backtrack_iters):
            new_flat = params_flat - coeff * full_step
            self._set_flat_params(new_flat)
            new_loss = loss_fn().item()
            kl       = kl_fn().item()
            if new_loss < old_loss and kl <= self.cfg.max_kl:
                return  # Accept
            coeff *= self.cfg.backtrack_coeff
        # Revert
        self._set_flat_params(params_flat)

    # ------------------------------------------------------------------

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.model.get_action_probs(obs).squeeze(0).cpu().numpy()
        return probs.astype(np.float32)

    def predict(self, state: np.ndarray) -> int:
        return int(np.argmax(self.get_action_probs(state)))

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict()}, path)
        log.info(f"TRPO saved to {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        log.info(f"TRPO loaded from {path}")
