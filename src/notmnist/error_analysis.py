"""Error analysis on test_clean: confusions, confident errors, coursework indices."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from notmnist import CLASS_NAMES
from notmnist.data import make_splits
from notmnist.utils import REPO_ROOT

COURSEWORK_IDX = (148, 446, 1201)
PLACEHOLDER = "_To be written from the figures after final runs._"
L = CLASS_NAMES


def top_confusions(y, pred, k: int = 10) -> list[tuple[int, int, int]]:
    """(true, pred, count) off-diagonal, sorted by count desc then (true, pred) asc."""
    c = Counter((int(t), int(p)) for t, p in zip(y, pred) if t != p)
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(t, p, n) for (t, p), n in items[:k]]


def _is_smoke(name: str) -> bool:
    return re.search(r"_sub\d+$", name) is not None


def resolve_run(name: str) -> Path:
    d = REPO_ROOT / "runs" / name
    if not (d / "test_probs.npy").exists():
        raise SystemExit(f"run '{name}' not found or has no test_probs.npy at {d}")
    return d


def default_best() -> str:
    path = REPO_ROOT / "reports" / "results.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; run `python -m notmnist.compare` first")
    rows = [r for r in json.loads(path.read_text()) if not r["run"].startswith("legacy_keras_")]
    if not rows:
        raise SystemExit("no non-legacy runs in results.json")
    return rows[-1]["run"]


def plot_confusion(y, pred, run, out):
    cm = np.zeros((10, 10))
    for t, p in zip(y, pred):
        cm[t, p] += 1
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            v = cmn[i, j]
            ax.text(j, i, f"{v:.2f}" if v >= 0.005 else "", ha="center", va="center",
                    fontsize=8, color="white" if v > 0.5 else "black")
    ax.set_xticks(range(10), L)
    ax.set_yticks(range(10), L)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalised), test_clean: {run}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confident_errors(x, y, pred, probs, orig_idx, run, out, n=25):
    wrong = np.flatnonzero(y != pred)
    conf = probs[wrong, pred[wrong]]
    sel = wrong[np.argsort(-conf)[:n]]
    fig, axes = plt.subplots(5, 5, figsize=(10, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), sel):
        ax.imshow(x[i], cmap="gray")
        ax.set_title(f"{orig_idx[i]}: {L[y[i]]}→{L[pred[i]]} ({probs[i, pred[i]]:.2f})", fontsize=9)
    fig.suptitle(f"Most confident errors, test_clean: {run}  (idx: true→pred (p))")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confidence_hist(y, pred, probs, run, out):
    mx = probs.max(1)
    ok = pred == y
    bins = np.linspace(0.1, 1, 28)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mx[ok], bins=bins, alpha=0.7, label=f"correct (n={ok.sum()})")
    ax.hist(mx[~ok], bins=bins, alpha=0.7, label=f"wrong (n={(~ok).sum()})")
    ax.set_yscale("log")
    ax.set_xlabel("max softmax probability")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Confidence: correct vs wrong, test_clean: {run}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def coursework_table(x_off, y_off, runs, out):
    """runs: {name: official probs}. Returns {run: {idx: (letter, conf)}} and saves a figure."""
    tab = {r: {i: (L[int(p[i].argmax())], float(p[i].max())) for i in COURSEWORK_IDX} for r, p in runs.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3 + 0.32 * len(runs)))
    for ax, i in zip(axes, COURSEWORK_IDX):
        ax.imshow(x_off[i], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"official idx {i} (true {L[y_off[i]]})")
        ax.set_xlabel("\n".join(f"{r}: {tab[r][i][0]} ({tab[r][i][1]:.2f})" for r in runs), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return tab


def _md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def error_deltas(y, pred_base, pred_best) -> dict:
    """fixed: baseline wrong & best right; introduced: baseline right & best wrong."""
    y, pb, pn = np.asarray(y), np.asarray(pred_base), np.asarray(pred_best)
    return {"fixed": int(((pb != y) & (pn == y)).sum()),
            "introduced": int(((pb == y) & (pn != y)).sum()),
            "n_err_base": int((pb != y).sum()), "n_err_best": int((pn != y).sum())}


def build_markdown(best, base, y, pred_base, pred_best, n, f1_base, f1_best, tab):
    d = error_deltas(y, pred_base, pred_best)
    err = d["n_err_best"]
    pairs = [(L[t], L[p], c, f"{100 * c / err:.1f}%") for t, p, c in top_confusions(y, pred_best)]
    f1 = [(c, f"{f1_base[c]['f1']:.3f}", f"{f1_best[c]['f1']:.3f}",
           f"{f1_best[c]['f1'] - f1_base[c]['f1']:+.3f}") for c in L]
    cw = [(r, *[f"{tab[r][i][0]} ({tab[r][i][1]:.2f})" for i in COURSEWORK_IDX]) for r in tab]
    return "\n\n".join([
        "# Error analysis",
        f"Best model: `{best}`; baseline: `{base}`; test_clean n = {n}. Best-model errors: {err}.",
        f"## Top-10 confused pairs ({best})",
        _md_table(["true", "pred", "count", "% of errors"], pairs),
        f"## Per-class F1 ({base} vs {best})",
        _md_table(["class", base, best, "delta (best - baseline)"], f1),
        "## Fixed / introduced errors",
        f"Baseline `{base}` -> best `{best}`: fixed {d['fixed']} (baseline wrong, best right), "
        f"introduced {d['introduced']} (baseline right, best wrong).",
        "## Coursework indices (official test set)",
        _md_table(["run", *[f"idx {i}" for i in COURSEWORK_IDX]], cw),
        "## Figures",
        f"![confusion](figures/confusion_{best}.png)\n![confident errors](figures/confident_errors_{best}.png)\n"
        f"![confidence](figures/confidence_hist_{best}.png)\n![coursework](figures/coursework_indices.png)",
        "## Observations", PLACEHOLDER, ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best")
    ap.add_argument("--baseline", default="mlp_baseline_s42")
    a = ap.parse_args()
    best = a.best or default_best()
    base = a.baseline
    dbest, dbase = resolve_run(best), resolve_run(base)

    sp = make_splits(seed=42)
    mask = sp.test_clean_mask
    orig_idx = np.flatnonzero(mask)
    x, y = sp.x_test[mask], sp.y_test[mask]
    pr_best = np.load(dbest / "test_probs.npy")
    pr_base = np.load(dbase / "test_probs.npy")
    pn, pb = pr_best[mask].argmax(1), pr_base[mask].argmax(1)
    probs = pr_best[mask]

    figs = REPO_ROOT / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot_confusion(y, pn, best, figs / f"confusion_{best}.png")
    plot_confident_errors(x, y, pn, probs, orig_idx, best, figs / f"confident_errors_{best}.png")
    plot_confidence_hist(y, pn, probs, best, figs / f"confidence_hist_{best}.png")

    cw_runs = {d.name: np.load(d / "test_probs.npy") for d in sorted((REPO_ROOT / "runs").iterdir())
               if d.is_dir() and not _is_smoke(d.name) and (d / "test_probs.npy").exists()}
    tab = coursework_table(sp.x_test, sp.y_test, cw_runs, figs / "coursework_indices.png")

    f1b = json.loads((dbase / "metrics.json").read_text())["test_clean"]["per_class"]
    f1n = json.loads((dbest / "metrics.json").read_text())["test_clean"]["per_class"]
    md = build_markdown(best, base, y, pred_base=pb, pred_best=pn, n=len(y),
                        f1_base=f1b, f1_best=f1n, tab=tab)
    (REPO_ROOT / "reports" / "error_analysis.md").write_text(md)
    print(f"wrote {REPO_ROOT / 'reports' / 'error_analysis.md'} (best={best}, baseline={base}, n={len(y)})")


if __name__ == "__main__":
    main()
