from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from utils.dataset import PolypDataset
from utils.transforms import get_train_transform, get_val_transform


@dataclass
class PolypDataConfig:
    image_size: int = 256
    batch_size: int = 4
    seed: int = 42
    num_workers: int = 0
    augment_train: bool = True
    train_ratio: float = 0.7


def build_kvasir_dataloaders(project_root, cfg: PolypDataConfig):
    data_root = Path(project_root) / "data" / "Kvasir-SEG"
    image_dir = str(data_root / "images")
    mask_dir  = str(data_root / "masks")

    # Determine split indices deterministically
    n = len(PolypDataset(image_dir, mask_dir))
    n_train = int(n * cfg.train_ratio)
    n_test  = n - n_train

    generator = torch.Generator().manual_seed(cfg.seed)
    indices   = torch.randperm(n, generator=generator).tolist()
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    train_transform = get_train_transform(cfg.image_size) if cfg.augment_train else get_val_transform(cfg.image_size)
    val_transform   = get_val_transform(cfg.image_size)

    train_ds = PolypDataset(image_dir, mask_dir, transform=train_transform)
    test_ds  = PolypDataset(image_dir, mask_dir, transform=val_transform)

    loaders = {
        "train": DataLoader(Subset(train_ds, train_idx), batch_size=cfg.batch_size,
                            shuffle=True,  num_workers=cfg.num_workers, pin_memory=True),
        "test":  DataLoader(Subset(test_ds,  test_idx),  batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers, pin_memory=True),
    }
    split_summary = {"train": n_train, "test": n_test}
    return loaders, split_summary, data_root
