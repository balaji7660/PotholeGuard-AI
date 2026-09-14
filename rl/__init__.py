"""
rl/__init__.py
Lazy imports to avoid hard dependency on gymnasium/stable-baselines3
when only individual submodules (reward, ensemble) are needed.
"""
# Direct submodule imports (no gymnasium dependency)
from rl.reward import RewardCalculator
from rl.ensemble import SoftVotingEnsemble

# Lazy agent/environment imports (require gymnasium + stable-baselines3)
def _lazy_import(name):
    import importlib
    return importlib.import_module(f"rl.{name}")

def get_environment():  return _lazy_import("environment").PotholeAvoidanceEnv
def get_ppo():          return _lazy_import("ppo_agent").PPOAgent
def get_a2c():          return _lazy_import("a2c_agent").A2CAgent
def get_trpo():         return _lazy_import("trpo_agent").TRPOAgent
def get_recurrent_ppo():return _lazy_import("recurrent_ppo").RecurrentPPOAgent

__all__ = [
    "RewardCalculator", "SoftVotingEnsemble",
    "get_environment", "get_ppo", "get_a2c", "get_trpo", "get_recurrent_ppo",
]

