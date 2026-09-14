"""
scripts/train_perception.py
Train the TransUNet perception model.

Usage:
    python scripts/train_perception.py \
        --model_cfg  configs/model_config.yaml \
        --train_cfg  configs/training_config.yaml \
        --dataset_cfg configs/dataset_config.yaml \
        [--ckpt_path checkpoints/perception/checkpoint_best.pt]
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from data.dataloader import build_dataloaders
from models.transunet import TransUNet
from perception.trainer import PerceptionTrainer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg",   default="configs/model_config.yaml")
    parser.add_argument("--train_cfg",   default="configs/training_config.yaml")
    parser.add_argument("--dataset_cfg", default="configs/dataset_config.yaml")
    parser.add_argument("--ckpt_path",   default=None)
    parser.add_argument("--seed",        type=int, default=None)
    args = parser.parse_args()

    model_cfg   = OmegaConf.load(args.model_cfg)
    train_cfg   = OmegaConf.load(args.train_cfg)
    dataset_cfg = OmegaConf.load(args.dataset_cfg)

    seed = args.seed or train_cfg.training.seed
    torch.manual_seed(seed)

    dev_str = train_cfg.training.device
    if dev_str == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_str)
    log.info(f"Device: {device}")

    # Data
    loaders = build_dataloaders(dataset_cfg, train_cfg)
    log.info(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)}")

    # Model
    model = TransUNet(model_cfg)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ckpt_path:
        epoch = PerceptionTrainer.load_checkpoint(model, args.ckpt_path, device)
        log.info(f"Resumed from epoch {epoch}")

    # Train
    trainer = PerceptionTrainer(model, loaders["train"], loaders["val"], train_cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
