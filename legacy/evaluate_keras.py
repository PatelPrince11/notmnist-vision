"""Evaluate the original Keras models on the shared notMNIST test split.

Run with the legacy env: env/bin/python legacy/evaluate_keras.py
Writes runs/legacy_keras_{mlp,cnn}/{test_probs.npy,metrics.json}.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from notmnist.data import make_splits  # noqa: E402
from notmnist.metrics import compute_metrics  # noqa: E402
import keras  # noqa: E402

MODELS = [
    ("legacy_keras_mlp", "keras_mlp", "notMNIST-Partial.keras"),
    ("legacy_keras_cnn", "keras_cnn", "notMNIST-Complete.keras"),
]


def main():
    splits = make_splits(seed=42)
    x = (splits.x_test.astype("float32") / 255.0)[..., None]
    y = splits.y_test
    mask = splits.test_clean_mask
    for run, model_name, fname in MODELS:
        model = keras.models.load_model(ROOT / "legacy" / "keras_part1" / fname)
        probs = model.predict(x, batch_size=1024, verbose=0).astype(np.float32)
        out = ROOT / "runs" / run
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "test_probs.npy", probs)
        result = {
            "run": run,
            "model_name": model_name,
            "n_params": int(model.count_params()),
            "val": None,
            "test_clean": compute_metrics(probs[mask], y[mask]),
            "test_official": compute_metrics(probs, y),
        }
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        print(f"{run}: n_params={result['n_params']} "
              f"test_official_acc={result['test_official']['accuracy']:.4f} "
              f"test_clean_acc={result['test_clean']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
