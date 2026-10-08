"""Shared helpers. Importable without torch; torch is imported lazily inside functions."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/notMNIST.npz"


def seed_everything(seed: int):
    """Seed random, numpy and torch; return a seeded torch.Generator for DataLoaders."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_device(prefer: str | None = None):
    """Return `prefer` if given, else the first available of mps, cuda, cpu."""
    import torch

    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
