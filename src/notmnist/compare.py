"""Compare evaluated runs: results table + McNemar tests on test_clean."""
import json
import math
import re
from pathlib import Path

import numpy as np

from notmnist.data import make_splits
from notmnist.utils import REPO_ROOT

LADDER = [
    ("legacy_keras_mlp", "mlp_baseline_s42"),
    ("mlp_baseline_s42", "cnn_baseline_s42"),
    ("cnn_baseline_s42", "cnn_improved_s42"),
    ("cnn_improved_s42", "resnet18_finetune_s42"),
    ("resnet18_probe_s42", "resnet18_finetune_s42"),
    ("resnet18_scratch_s42", "resnet18_finetune_s42"),
]


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Two-sided exact McNemar. b01: a wrong, b right. b10: a right, b wrong."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p_value": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5**n)
    return {"b01": b01, "b10": b10, "p_value": p}


def _is_smoke(name: str) -> bool:
    return re.search(r"_sub\d+$", name) is not None


def build_rows(runs_dir: Path) -> list[dict]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")
    rows = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or _is_smoke(d.name) or not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        tc = m["test_clean"]
        rows.append({
            "run": m["run"], "model_name": m["model_name"], "n_params": m["n_params"],
            "val_acc": m["val"]["accuracy"] if m.get("val") else None,
            "test_clean_acc": tc["accuracy"], "test_clean_ci95": list(tc["accuracy_ci95"]),
            "test_official_acc": m["test_official"]["accuracy"],
            "macro_f1": tc["macro_f1"], "nll": tc["nll"], "ece": tc["ece"],
        })
    rows.sort(key=lambda r: r["test_clean_acc"])
    return rows


def ladder_pairs(runs_dir: Path, y_test: np.ndarray, mask: np.ndarray) -> list[dict]:
    runs_dir = Path(runs_dir)
    out = []
    for a, b in LADDER:
        pa, pb = runs_dir / a / "test_probs.npy", runs_dir / b / "test_probs.npy"
        if not (pa.exists() and pb.exists() and (runs_dir / a / "metrics.json").exists()
                and (runs_dir / b / "metrics.json").exists()):
            continue
        ca = np.load(pa).argmax(1)[mask] == y_test[mask]
        cb = np.load(pb).argmax(1)[mask] == y_test[mask]
        out.append({"a": a, "b": b, **mcnemar_exact(ca, cb)})
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render_markdown(rows: list[dict], pairs: list[dict], n_clean: int, n_official: int) -> str:
    L = [
        "# Results",
        "",
        f"Data split: test_clean n={n_clean}, test_official n={n_official}. Numbers come from "
        "`runs/*/metrics.json` produced by `python -m notmnist.evaluate`.",
        "",
        "| run | params | val acc | test_clean acc [95% CI] | test_official acc | macro-F1 | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lo, hi = r["test_clean_ci95"]
        val = "—" if r["val_acc"] is None else _pct(r["val_acc"])
        L.append(f"| {r['run']} | {r['n_params']:,} | {val} | {_pct(r['test_clean_acc'])} "
                 f"[{100 * lo:.2f}–{100 * hi:.2f}] | {_pct(r['test_official_acc'])} "
                 f"| {r['macro_f1']:.4f} | {r['nll']:.4f} | {r['ece']:.4f} |")
    L += ["", "## McNemar ladder (test_clean)", "",
          "b01 = a wrong, b right; b10 = a right, b wrong; p-value is a two-sided exact binomial test.",
          "", "| a | b | b01 | b10 | p-value |", "|---|---|---:|---:|---:|"]
    for p in pairs:
        L.append(f"| {p['a']} | {p['b']} | {p['b01']} | {p['b10']} | {p['p_value']:.4g} |")
    return "\n".join(L) + "\n"


def main() -> None:
    runs_dir, out_dir = REPO_ROOT / "runs", REPO_ROOT / "reports"
    splits = make_splits(seed=42)
    mask = splits.test_clean_mask
    rows = build_rows(runs_dir)
    pairs = ladder_pairs(runs_dir, splits.y_test, mask)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "mcnemar.json").write_text(json.dumps(pairs, indent=2))
    (out_dir / "results.md").write_text(
        render_markdown(rows, pairs, int(mask.sum()), len(splits.y_test)))
    print(f"wrote {len(rows)} rows, {len(pairs)} pairs to {out_dir}")


if __name__ == "__main__":
    main()
