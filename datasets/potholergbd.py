"""
PotholeRGBD Dataset Loader.
Loads RGB road images, binary masks, and synchronized depth maps.
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

class PotholeRGBDDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []

        if os.path.exists(root_dir):
            rgb_dir = os.path.join(root_dir, "rgb")
            mask_dir = os.path.join(root_dir, "masks")
            depth_dir = os.path.join(root_dir, "depth")
            
            if os.path.exists(rgb_dir):
                valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                rgb_paths = []
                for ext in valid_exts:
                    rgb_paths.extend(glob.glob(os.path.join(rgb_dir, ext)))

                for p in sorted(rgb_paths):
                    base = os.path.splitext(os.path.basename(p))[0]
                    mask_path = os.path.join(mask_dir, f"{base}.png")
                    depth_path = os.path.join(depth_dir, f"{base}.png")
                    if not os.path.exists(depth_path):
                        depth_path = os.path.join(depth_dir, f"{base}.npy")

                    if os.path.exists(mask_path) and os.path.exists(depth_path):
                        self.samples.append({
                            "image_path": p,
                            "mask_path": mask_path,
                            "depth_path": depth_path,
                            "has_gt_depth": True
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"])
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        depth_p = sample["depth_path"]
        if depth_p.endswith(".npy"):
            depth = np.load(depth_p).astype(np.float32)
        else:
            depth_raw = cv2.imread(depth_p, cv2.IMREAD_UNCHANGED)
            if depth_raw is not None:
                depth = depth_raw.astype(np.float32) / (np.max(depth_raw) + 1e-6)
            else:
                depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform:
            img_t, mask_t, depth_t = self.transform(image, mask, depth=depth)
        else:
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
            depth_t = torch.from_numpy(depth).unsqueeze(0)

        return {
            "image": img_t,
            "mask": mask_t,
            "depth": depth_t,
            "has_gt_depth": torch.tensor(1, dtype=torch.uint8),
            "meta": sample
        }
