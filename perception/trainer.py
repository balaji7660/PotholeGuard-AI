"""
perception/trainer.py
Training and validation loop for the multi-task TransUNet.

Features:
  - AMP (automatic mixed precision)
  - Gradient clipping
  - LR scheduling with warmup
  - TensorBoard logging
  - Best-checkpoint saving
  - Early stopping
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.transunet import TransUNet
from models.loss import MultiTaskLoss
from perception.evaluator import PerceptionEvaluator

log = logging.getLogger(__name__)


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int) -> None:
        import math
        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda)


class PerceptionTrainer:
    """
    Trains the TransUNet perception model.

    Parameters
    ----------
    model : TransUNet
    train_loader : DataLoader
    val_loader   : DataLoader
    train_cfg    : DictConfig  (training_config.yaml)
    device       : torch.device
    """

    def __init__(
        self,
        model: TransUNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = train_cfg.training
        self.device       = device

        # Loss
        self.criterion = MultiTaskLoss(train_cfg.loss)

        # Optimiser
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.cfg.scheduler.warmup_epochs,
            total_epochs=self.cfg.epochs,
        )

        # AMP
        self.scaler = GradScaler(enabled=self.cfg.amp)

        # Checkpointing
        self.ckpt_dir = Path(self.cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        log_dir = Path(train_cfg.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir)) if train_cfg.logging.tensorboard else None

        # Early stopping state
        self.best_val_dice = -1.0
        self.patience_counter = 0
        es_cfg = self.cfg.early_stopping
        self.early_stopping_patience = es_cfg.patience if es_cfg.enabled else None

        # Evaluator
        self.evaluator = PerceptionEvaluator()

    # ------------------------------------------------------------------
    # Optimiser factory
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.cfg.optimizer
        params  = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_cfg.type == "adamw":
            return torch.optim.AdamW(
                params,
                lr=opt_cfg.lr,
                betas=opt_cfg.betas,
                weight_decay=opt_cfg.weight_decay,
            )
        elif opt_cfg.type == "adam":
            return torch.optim.Adam(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.type == "sgd":
            return torch.optim.SGD(
                params, lr=opt_cfg.lr, momentum=0.9, weight_decay=opt_cfg.weight_decay
            )
        raise ValueError(f"Unknown optimizer: {opt_cfg.type}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        log.info("Starting perception training.")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_losses = self._train_epoch(epoch)
            val_metrics  = self._validate(epoch)
            self.scheduler.step()

            elapsed = time.time() - t0
            log.info(
                f"Epoch {epoch}/{self.cfg.epochs}  "
                f"train_loss={train_losses['total']:.4f}  "
                f"val_dice={val_metrics['dice']:.4f}  "
                f"val_iou={val_metrics['iou']:.4f}  "
                f"val_depth_rmse={val_metrics.get('depth_rmse', 0.0):.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                f"time={elapsed:.1f}s"
            )

            # TensorBoard
            if self.writer:
                for k, v in train_losses.items():
                    self.writer.add_scalar(f"train/{k}_loss", v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            # Save checkpoint every N epochs
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, val_metrics)

            # Best checkpoint
            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self._save_checkpoint(epoch, val_metrics, tag="best")
                self.patience_counter = 0
                log.info(f"  ✓ New best val_dice: {self.best_val_dice:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if (
                self.early_stopping_patience is not None
                and self.patience_counter >= self.early_stopping_patience
            ):
                log.info(f"Early stopping triggered after {epoch} epochs.")
                break

        if self.writer:
            self.writer.close()
        log.info("Training complete.")

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_losses: Dict[str, float] = {"total": 0, "seg": 0, "depth": 0, "uncertainty": 0}
        n_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, 1):
            images = batch["image"].to(self.device, non_blocking=True)
            masks  = batch["mask"].to(self.device, non_blocking=True)
            depths = batch["depth"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg.amp):
                outputs = self.model(images)
                losses  = self.criterion(
                    outputs,
                    {"mask": masks, "depth": depths},
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in total_losses:
                total_losses[k] += losses[k].item()

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        all_metrics: list[Dict[str, float]] = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks  = batch["mask"].to(self.device, non_blocking=True)
            depths = batch["depth"].to(self.device, non_blocking=True)

            outputs = self.model(images)
            metrics = self.evaluator.compute_batch(
                pred_seg=outputs["segmentation"],
                gt_seg=masks,
                pred_dep=outputs["depth"],
                gt_dep=depths,
            )
            all_metrics.append(metrics)

        # Average across batches
        averaged: Dict[str, float] = {}
        for key in all_metrics[0]:
            averaged[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        return averaged

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], tag: str = "") -> None:
        fname = f"checkpoint_epoch{epoch}.pt" if not tag else f"checkpoint_{tag}.pt"
        path  = self.ckpt_dir / fname
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "metrics":         metrics,
                "best_val_dice":   self.best_val_dice,
            },
            path,
        )
        log.info(f"  Checkpoint saved → {path}")

    @classmethod
    def load_checkpoint(cls, model: TransUNet, ckpt_path: str, device: torch.device) -> int:
        """Load model weights from checkpoint. Returns the epoch number."""
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return ckpt.get("epoch", 0)
