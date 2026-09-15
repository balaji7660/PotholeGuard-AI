"""
Unified Dataset combining Pothole-600 (segmentation) and PotholeRGBD (RGB-D) datasets.
Gracefully handles missing datasets or missing ground-truth depth channels.
"""
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, Callable
from .pothole600 import Pothole600Dataset
from .potholergbd import PotholeRGBDDataset

class UnifiedPotholeDataset(Dataset):
    def __init__(
        self,
        pothole600_dir: Optional[str] = None,
        potholergbd_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        datasets = []
        if pothole600_dir:
            ds600 = Pothole600Dataset(pothole600_dir, split=split, transform=transform)
            if len(ds600) > 0:
                datasets.append(ds600)
        
        if potholergbd_dir:
            ds_rgbd = PotholeRGBDDataset(potholergbd_dir, split=split, transform=transform)
            if len(ds_rgbd) > 0:
                datasets.append(ds_rgbd)

        self.combined = ConcatDataset(datasets) if len(datasets) > 0 else None

    def __len__(self) -> int:
        return len(self.combined) if self.combined is not None else 0

    def __getitem__(self, idx: int):
        if self.combined is None:
            raise IndexError("Unified dataset is empty.")
        return self.combined[idx]
