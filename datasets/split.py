"""
Reproducible Dataset Splitting Utility for PotholeGuard-AI.
Splits datasets into reproducible Train (70%), Validation (15%), and Test (15%) splits with fixed seeds.
"""
import random
from typing import List, Dict, Tuple

def create_reproducible_splits(
    samples: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split sample metadata list into train, val, and test splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]

    return train_set, val_set, test_set
