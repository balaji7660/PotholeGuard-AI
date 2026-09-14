"""
data/__init__.py
Exports the unified dataset interface and DataLoader factory.
"""
from data.dataset import PotholeDataset, DatasetMode
from data.dataloader import build_dataloaders
from data.pseudo_depth import PseudoDepthGenerator

__all__ = [
    "PotholeDataset",
    "DatasetMode",
    "build_dataloaders",
    "PseudoDepthGenerator",
]
