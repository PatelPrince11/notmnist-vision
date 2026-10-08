# CLAUDE.md

## Project context

This repo started as a CPSC 433 assignment: Keras MLP and CNN models on notMNIST (28×28 font-rendered letters A–J),
plus an unrelated tabular NFL-draft model. It is being turned into a portfolio-quality **PyTorch computer-vision
project**: a leak-free data pipeline, a model ladder (MLP → CNN → improved CNN → ResNet-18 transfer learning),
rigorous evaluation, error analysis and a small inference path.

- Design and audited facts: `PROJECT_GUIDE.md` (read §2, §5 and §7 before starting work)
- Task-by-task plan: `docs/superpowers/plans/2026-10-07-notmnist-portfolio.md`
- Original coursework lives in `legacy/` and is **read-only**: don't refactor it, don't "fix" it.

Audited baseline facts (measured 2026-10-07 from the committed `.keras` files, don't re-derive):
Keras MLP 93.10 % / CNN 95.12 % on the official 10k test set, and 92.82 % / 94.91 % on the de-duplicated
9,487 subset. 513 test images duplicate training images. Train contains 1,761 duplicates.

## Commands

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"     # setup
.venv/bin/pytest -q                                             # tests
.venv/bin/python -m notmnist.train --preset cnn_improved --seed 42
.venv/bin/python -m notmnist.evaluate runs/cnn_improved_s42
.venv/bin/python -m notmnist.compare
.venv/bin/python -m notmnist.error_analysis
env/bin/python legacy/evaluate_keras.py                          # legacy TF env only
```

## Technical principles

- Prefer simple, maintainable implementations. Plain PyTorch loops, no Lightning, no Hydra.
- Don't add dependencies beyond `pyproject.toml` without asking first.
- Reuse existing working code when practical. Port architectures exactly; don't redesign them silently.
- **Never fabricate metrics, never invent benchmark results, never claim an experiment succeeded without running it.**
  Every number written to README/reports must come from a `metrics.json` that a command produced in this repo.
  If a run wasn't executed, write "not run".
- Preserve reproducibility: seeded splits, `seed_everything`, the resolved config saved in every run directory.
  Bitwise determinism on MPS isn't guaranteed, so say so instead of claiming it.
- Keep experimentation (`train`, `evaluate`, `compare`, `error_analysis`) separate from production
  inference (`inference.py`, `predict.py`, `api/`). Inference must not import training code.

## Development workflow

1. Inspect the relevant code before modifying it.
2. Explain the proposed change in a sentence or two.
3. Make small, focused changes, one plan task at a time.
4. Run the relevant tests or experiments.
5. Verify the outputs (open the file, read the number, look at the figure).
6. Report actual results, including failures, regressions and surprises.
7. Update `PROJECT_GUIDE.md` / `README.md` when behaviour or results change.

## Deep-learning rules

- **Train / val / test are distinct.** The val split drives early stopping, checkpoint selection and every
  hyperparameter decision. The test split is read only by `notmnist.evaluate`.
- Avoid leakage: duplicates are removed from train before splitting. The primary metric is on
  **test_clean** (test images not present in train). Also report test_official for continuity.
- Record hyperparameters: everything lives in `TrainConfig` and gets written to `config.json`.
- Record actual metrics: `metrics.json` per run. Never hand-edit it.
- Compare against baselines: every new model appears in `reports/results.md` next to the MLP/CNN
  baselines, the legacy Keras rows, and a McNemar p-value against its predecessor.
- Use appropriate metrics. Classes are balanced, so accuracy is meaningful, but always report macro-F1,
  per-class F1, NLL and ECE too. Don't optimise for accuracy alone.
- Checkpoints store `model_name`, `config` and `class_names`, so a `.pt` file is loadable without the preset table.
- Inference is reproducible: `model.eval()`, `torch.inference_mode()`, one shared `preprocess_image`.
- Model input contract: `(B, 1, 28, 28)` float in [0, 1], light glyph on dark background. Model-specific
  resizing/normalisation lives inside the model's `forward`.
- When a frozen backbone is trained, its BatchNorm layers stay in eval mode.

## Scope control

- The one-day Phase 1 MVP (`PROJECT_GUIDE.md` §7) comes first. Don't start Phase 2 until Phase 1 is
  verified, and don't start Phase 3 items unless explicitly asked.
- If something will take significantly longer than its plan estimate (more than about 30 minutes over),
  stop and explain the tradeoff before continuing.
- No UI beyond the CLI (and the Phase 2 API). No microservices. No abstractions with a single implementation.

## Housekeeping

- `runs/` is gitignored. Commit `reports/` and the single demo checkpoint in `models/`.
- `env/` is the legacy TF venv (Python 3.13, TF 2.20). Use it only for `legacy/evaluate_keras.py`.
- Course-template files in `legacy/keras_part1/` credit their original author (Jonathan Hudson). Keep that attribution.
  Don't put the student ID in new files.
- Low disk space on this machine (~20 GB free). Avoid unnecessary Docker images and checkpoint copies.
