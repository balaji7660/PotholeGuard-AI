"""
scripts/train_rl.py
Train individual RL agents or all four.

Usage:
    python scripts/train_rl.py --agent ppo
    python scripts/train_rl.py --agent all
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from rl.environment import PotholeAvoidanceEnv
from rl.ppo_agent import PPOAgent
from rl.a2c_agent import A2CAgent
from rl.trpo_agent import TRPOAgent
from rl.recurrent_ppo import RecurrentPPOAgent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AGENT_CLASSES = {
    "ppo":           PPOAgent,
    "a2c":           A2CAgent,
    "trpo":          TRPOAgent,
    "recurrent_ppo": RecurrentPPOAgent,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",       default="ppo",
                        choices=list(AGENT_CLASSES.keys()) + ["all"])
    parser.add_argument("--rl_cfg",      default="configs/rl_config.yaml")
    parser.add_argument("--severity_cfg",default="configs/severity_config.yaml")
    parser.add_argument("--srl_cfg",     default="configs/srl_config.yaml")
    parser.add_argument("--device",      default="auto")
    args = parser.parse_args()

    rl_cfg       = OmegaConf.load(args.rl_cfg)
    severity_cfg = OmegaConf.load(args.severity_cfg)
    srl_cfg      = OmegaConf.load(args.srl_cfg)

    def make_env():
        return PotholeAvoidanceEnv(rl_cfg, severity_cfg, srl_cfg)

    agents_to_train = list(AGENT_CLASSES.keys()) if args.agent == "all" else [args.agent]

    for agent_name in agents_to_train:
        log.info(f"Training {agent_name.upper()}...")
        env   = make_env()
        cls   = AGENT_CLASSES[agent_name]
        agent = cls(env, rl_cfg, device=args.device)
        agent.train()
        log.info(f"  {agent_name.upper()} training complete.")


if __name__ == "__main__":
    main()
