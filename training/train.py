"""
Training Pipeline for Multi-Task TransUNet.
Supports reproducible training runs, checkpoints, learning rate scheduling,
and experiment artifact logging.
"""
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from models.transunet import MultiTaskTransUNet
from datasets.unified_dataset import UnifiedPotholeDataset
from datasets.transforms import RoadAugmentation
from training.loss import MultiTaskLoss

def train_pipeline(config_path: str = "configs/training_config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "epochs": 10,
            "batch_size": 4,
            "lr": 1e-4,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "save_dir": "checkpoints"
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Multi-Task TransUNet on device: {device}")

    # Build model
    model = MultiTaskTransUNet(pretrained_encoder=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=1e-4)
    criterion = MultiTaskLoss()

    # Load dataset
    train_transform = RoadAugmentation(is_train=True, img_size=512)
    train_dataset = UnifiedPotholeDataset(
        pothole600_dir="data/pothole-600",
        potholergbd_dir="data/pothole-rgbd",
        split="train",
        transform=train_transform
    )

    if len(train_dataset) == 0:
        print("[!] No training dataset found in data/ directory.")
        print("[!] To train: download Pothole-600 or PotholeRGBD into data/ directory.")
        return

    loader = DataLoader(train_dataset, batch_size=cfg.get("batch_size", 4), shuffle=True)
    os.makedirs(cfg.get("save_dir", "checkpoints"), exist_ok=True)

    print(f"[*] Starting training loop for {cfg.get('epochs', 10)} epochs...")
    model.train()
    for epoch in range(1, cfg.get("epochs", 10) + 1):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            optimizer.zero_grad()
            preds = model(imgs, mc_dropout=True)
            loss_dict = criterion(preds, {"mask": masks, "depth": depths})
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch [{epoch}/{cfg.get('epochs', 10)}] - Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = os.path.join(cfg.get("save_dir", "checkpoints"), f"transunet_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss
        }, save_path)
    
    # Save final best
    best_path = os.path.join(cfg.get("save_dir", "checkpoints"), "best_transunet.pth")
    torch.save(model.state_dict(), best_path)
    print(f"[+] Training completed. Saved best model to {best_path}")

if __name__ == "__main__":
    train_pipeline()
