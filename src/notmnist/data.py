"""Leak-free notMNIST data pipeline. Importable without torch (torch is imported lazily)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split

from notmnist.utils import DATA_PATH

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


def load_raw(path: Path = DATA_PATH) -> dict[str, np.ndarray]:
    """Load the npz into a plain dict of arrays."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def image_hashes(x: np.ndarray) -> np.ndarray:
    """MD5 hex digest of each image's bytes."""
    x = np.ascontiguousarray(x)
    return np.array([hashlib.md5(img.tobytes()).hexdigest() for img in x], dtype=object)


@dataclass
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_clean_mask: np.ndarray
    info: dict


def make_splits(seed: int = 42, val_frac: float = 0.1, path: Path = DATA_PATH) -> Splits:
    """Dedup train (keep first), stratified train/val split, and flag test images that leak from train."""
    raw = load_raw(path)
    x_tr, y_tr, x_te, y_te = raw["x_train"], raw["y_train"], raw["x_test"], raw["y_test"]

    train_hashes = image_hashes(x_tr)
    test_hashes = image_hashes(x_te)

    uniq, first_idx, inverse = np.unique(train_hashes, return_index=True, return_inverse=True)
    keep = np.sort(first_idx)

    # Duplicate groups whose members carry more than one distinct label.
    inverse = inverse.reshape(-1)
    n_conflicting = 0
    for g in range(len(uniq)):
        labels = y_tr[inverse == g]
        if len(labels) > 1 and len(np.unique(labels)) > 1:
            n_conflicting += 1

    x_d, y_d = x_tr[keep], y_tr[keep]
    x_train, x_val, y_train, y_val = train_test_split(
        x_d, y_d, test_size=val_frac, stratify=y_d, random_state=seed
    )

    test_clean_mask = ~np.isin(test_hashes, train_hashes)

    info = {
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "n_train_raw": int(len(x_tr)),
        "n_train_dedup": int(len(keep)),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_te)),
        "n_test_clean": int(test_clean_mask.sum()),
        "n_train_dup_conflicting_labels": int(n_conflicting),
    }
    return Splits(x_train, y_train, x_val, y_val, x_te, y_te, test_clean_mask, info)


def to_tensor(x: np.ndarray) -> "torch.Tensor":
    """uint8 (N,28,28) -> float32 (N,1,28,28) scaled to [0,1]."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(x)).float().div(255.0).unsqueeze(1)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    generator: "torch.Generator | None" = None,
) -> "DataLoader":
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(to_tensor(x), torch.from_numpy(np.ascontiguousarray(y)).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)
