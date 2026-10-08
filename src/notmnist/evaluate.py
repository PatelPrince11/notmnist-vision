"""Evaluate trained runs: write test_probs.npy and metrics.json into each run dir."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from notmnist.data import make_splits, to_tensor
from notmnist.metrics import compute_metrics
from notmnist.models import count_params, load_checkpoint
from notmnist.utils import get_device


def predict_probs(model, x_uint8: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(x_uint8), batch_size):
            xb = to_tensor(x_uint8[i:i + batch_size]).to(device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def evaluate_run(run_dir: Path, device=None) -> dict:
    run_dir = Path(run_dir)
    device = device or get_device()
    seed = json.loads((run_dir / "config.json").read_text())["config"]["seed"]
    model, ckpt = load_checkpoint(run_dir / "best.pt", device=device)
    splits = make_splits(seed=seed)
    val_probs = predict_probs(model, splits.x_val, device)
    test_probs = predict_probs(model, splits.x_test, device)
    np.save(run_dir / "test_probs.npy", test_probs)
    mask = splits.test_clean_mask
    result = {
        "run": run_dir.name,
        "model_name": ckpt["model_name"],
        "n_params": count_params(model),
        "val": compute_metrics(val_probs, splits.y_val),
        "test_clean": compute_metrics(test_probs[mask], splits.y_test[mask]),
        "test_official": compute_metrics(test_probs, splits.y_test),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = get_device(a.device)
    for d in a.run_dirs:
        r = evaluate_run(d, device)
        print(f"{r['run']}: val {r['val']['accuracy']:.4f} | test_clean {r['test_clean']['accuracy']:.4f} "
              f"| test_official {r['test_official']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
