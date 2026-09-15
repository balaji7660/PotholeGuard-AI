"""
Pothole-600 Dataset Loader.
Loads RGB road images and corresponding binary segmentation masks.
Does NOT fabricate depth; provides pseudo-depth flag if teacher pseudo-depth is used.
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

class Pothole600Dataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []

        if os.path.exists(root_dir):
            img_dir = os.path.join(root_dir, "images")
            mask_dir = os.path.join(root_dir, "masks")
            if os.path.exists(img_dir):
                valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                img_paths = []
                for ext in valid_exts:
                    img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
                
                for p in sorted(img_paths):
                    base = os.path.splitext(os.path.basename(p))[0]
                    mask_path = os.path.join(mask_dir, f"{base}.png")
                    if not os.path.exists(mask_path):
                        mask_path = os.path.join(mask_dir, f"{base}.jpg")
                    
                    if os.path.exists(mask_path):
                        self.samples.append({
                            "image_path": p,
                            "mask_path": mask_path,
                            "has_gt_depth": False
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

        if self.transform:
            img_t, mask_t, depth_t = self.transform(image, mask, depth=None)
        else:
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
            depth_t = torch.zeros_like(mask_t)

        return {
            "image": img_t,
            "mask": mask_t,
            "depth": depth_t,
            "has_gt_depth": torch.tensor(0, dtype=torch.uint8),
            "meta": sample
        }
