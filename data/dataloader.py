"""
data/dataloader.py
DataLoader factory: creates train/val/test DataLoaders from config.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.dataset import DatasetMode, PotholeDataset
from data.pseudo_depth import PseudoDepthGenerator


def build_dataloaders(
    dataset_cfg: DictConfig,
    training_cfg: DictConfig,
    pseudo_gen: Optional[PseudoDepthGenerator] = None,
) -> Dict[str, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    Parameters
    ----------
    dataset_cfg : DictConfig
        Loaded from configs/dataset_config.yaml.
    training_cfg : DictConfig
        Loaded from configs/training_config.yaml.
    pseudo_gen : PseudoDepthGenerator | None
        Optional shared MiDaS generator (avoids loading it multiple times).

    Returns
    -------
    dict with keys "train", "val", "test"  →  DataLoader instances.
    """
    # Share a single MiDaS instance across splits
    if pseudo_gen is None and not dataset_cfg.datasets[dataset_cfg.datasets.active].has_depth:
        pseudo_gen = PseudoDepthGenerator(dataset_cfg)

    num_workers: int = training_cfg.training.num_workers
    pin_memory:  bool = training_cfg.training.pin_memory

    loaders: Dict[str, DataLoader] = {}

    for mode in DatasetMode:
        is_train = mode == DatasetMode.TRAIN
        bs = training_cfg.training.batch_size if is_train else training_cfg.training.val_batch_size

        dataset = PotholeDataset(
            dataset_cfg=dataset_cfg,
            mode=mode,
            pseudo_depth_generator=pseudo_gen,
        )

        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
            persistent_workers=(num_workers > 0),
        )
        loaders[mode.value] = loader

    return loaders


def get_sample_batch(
    dataset_cfg: DictConfig,
    training_cfg: DictConfig,
    split: str = "val",
    n: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Utility: grab a small batch from a split for quick visualisation/debugging.

    Returns
    -------
    (images, masks, depths)  each of shape (n, C, H, W) / (n, 1, H, W).
    """
    loaders = build_dataloaders(dataset_cfg, training_cfg)
    loader  = loaders[split]
    batch   = next(iter(loader))
    images  = batch["image"][:n]
    masks   = batch["mask"][:n]
    depths  = batch["depth"][:n]
    return images, masks, depths
