"""
tests/test_ensemble.py
Unit tests for the soft-voting ensemble.
"""
import numpy as np
import pytest
from omegaconf import OmegaConf
from rl.ensemble import SoftVotingEnsemble

_RL_CFG = OmegaConf.create({
    "ensemble": {
        "weights": {"ppo": 0.25, "a2c": 0.25, "trpo": 0.25, "recurrent_ppo": 0.25},
        "adaptive_weights": False,
    }
})


class _MockAgent:
    def __init__(self, probs): self._probs = np.array(probs, dtype=np.float32)
    def get_action_probs(self, state): return self._probs.copy()


class TestSoftVotingEnsemble:
    def _make_ensemble(self, probs_dict):
        agents = {k: _MockAgent(v) for k, v in probs_dict.items()}
        return SoftVotingEnsemble(agents, _RL_CFG)

    def test_equal_weights_average(self):
        e = self._make_ensemble({
            "ppo":           [0.2, 0.6, 0.2],
            "a2c":           [0.2, 0.6, 0.2],
            "trpo":          [0.2, 0.6, 0.2],
            "recurrent_ppo": [0.2, 0.6, 0.2],
        })
        probs = e.get_action_probs(np.zeros(117))
        np.testing.assert_allclose(probs, [0.2, 0.6, 0.2], atol=1e-5)

    def test_sums_to_one(self):
        e = self._make_ensemble({
            "ppo":           [0.1, 0.7, 0.2],
            "a2c":           [0.3, 0.5, 0.2],
            "trpo":          [0.25, 0.55, 0.2],
            "recurrent_ppo": [0.2, 0.6, 0.2],
        })
        probs = e.get_action_probs(np.zeros(117))
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_argmax_action(self):
        e = self._make_ensemble({
            "ppo":           [0.0, 1.0, 0.0],
            "a2c":           [0.0, 1.0, 0.0],
            "trpo":          [0.0, 1.0, 0.0],
            "recurrent_ppo": [0.0, 1.0, 0.0],
        })
        assert e.select_action(np.zeros(117)) == 1

    def test_four_agents_required(self):
        e = self._make_ensemble({
            "ppo":           [0.2, 0.5, 0.3],
            "a2c":           [0.3, 0.4, 0.3],
            "trpo":          [0.1, 0.7, 0.2],
            "recurrent_ppo": [0.25, 0.5, 0.25],
        })
        assert len(e.agents) == 4
