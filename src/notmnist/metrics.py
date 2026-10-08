"""Classification metrics (numpy/sklearn only; deliberately no torch import)."""
import math

import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

from notmnist import CLASS_NAMES

N_CLASSES = len(CLASS_NAMES)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins over max-probability confidence; confidence 1.0 falls in the top bin."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)  # float32 softmax rows may not sum to exactly 1
    y = np.asarray(y)
    labels = list(range(N_CLASSES))
    pred = probs.argmax(axis=1)
    n = int(len(y))
    k = int((pred == y).sum())
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    present = s > 0  # macro averages only over classes that occur in y
    return {
        "n": n,
        "accuracy": k / n,
        "accuracy_ci95": [float(v) for v in wilson_ci(k, n)],
        "macro_precision": float(p[present].mean()),
        "macro_recall": float(r[present].mean()),
        "macro_f1": float(f[present].mean()),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "nll": float(log_loss(y, probs, labels=labels)),
        "ece": expected_calibration_error(probs, y),
    }
