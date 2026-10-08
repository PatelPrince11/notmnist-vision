# notMNIST Letter Recognition — Project Guide

> From a CPSC 433 (Fall 2025) assignment to a reproducible PyTorch computer-vision project.
> This guide is self-contained: you do not need the original course report.

---

## 1. Purpose

Classify 28×28 grayscale images of the letters **A–J** rendered in many different fonts
(the notMNIST dataset). The project's job is to *show deep-learning judgement*, not to chase
a leaderboard number:

- a **ladder of models** (MLP → CNN → improved CNN → ImageNet transfer learning), each compared on the same leak-free split;
- **honest evaluation** (validation split, de-duplicated test set, confidence intervals, significance tests, calibration);
- **error analysis** that explains *why* models fail (confusable letter pairs, label noise);
- a **small, correct inference path** (CLI, then optionally a FastAPI endpoint in Docker).

---

## 2. Current state (audited 2026-10-07)

All numbers in this section were **re-measured from the committed artifacts** during the audit,
not copied from the report.

**Results (2026-10-07):** Phase 1 is complete. Measured results, key findings and figures are in `README.md`; the full tables are in `reports/results.md` and `reports/error_analysis.md`.

### 2.1 Repository contents

```
433A1/                         git repo, remote github.com/PatelPrince11/433A1, branch main
├── PART 1/                    notMNIST image classification (Keras)
│   ├── notMNIST.npz           dataset, 55 MB, tracked in git
│   ├── notMNIST-Partial.py    trains the MLP  → notMNIST-Partial.keras
│   ├── notMNIST-Complete.py   trains the CNN  → notMNIST-Complete.keras
│   ├── predict.py             classify one 28×28 PNG, plot softmax bars        (course template)
│   ├── predict_test.py        browse test images interactively                 (course template)
│   ├── interactive.py         Tk canvas: draw a letter, classify it            (course template)
│   ├── grabimage.py           Tk canvas → image.png                            (course template)
│   ├── find_misclassified.py  lists MLP test errors (output pasted as comment)
│   ├── compare_partial_complete.py  MLP vs CNN softmax on 3 chosen indices
│   └── images/, image.png     4 screenshots + 1 drawn sample
├── PART 2/                    NFL-combine "drafted?" tabular classifier (Keras)
│   ├── CFLModel.py
│   └── draft.csv              671 rows, 12 features + label
├── draft_train.csv / draft_test.csv   split written by CFLModel.py (cwd-relative!)
├── 433A1Files/ + .zip         submission bundle (copies of the above)
└── env/                       1.7 GB venv (Py 3.13, TF 2.20, Keras 3.12) — untracked, not ignored
```

There is no README, requirements file, `.gitignore`, test suite, config system or notebook.
`.DS_Store` files are tracked. The working tree has uncommitted edits (header comments and a
retrained `notMNIST-Complete.keras`).

### 2.2 Dataset (Part 1)

| | Train | Test |
|---|---|---|
| Images | 60,000 × 28×28 uint8 | 10,000 × 28×28 uint8 |
| Classes | A–J, balanced (5,998–6,002 each) | balanced (997–1,005 each) |
| Polarity | white glyph on black background | same |

**Data-quality findings (new, not in the report):**
- **1,761 duplicate images inside the training set** (58,239 unique of 60,000).
- **513 test images (5.1 %) are byte-identical to a training image.** That's test-set leakage. The saved models score
  98–99 % on those 513 versus 93–95 % on the rest.
- There is **no validation split**. Every architecture/epoch decision was made by looking at test accuracy.

### 2.3 Existing models (Part 1)

| | MLP (“Partial”) | CNN (“Complete”) |
|---|---|---|
| Architecture | Flatten → Dense 256 ReLU → Dense 10 softmax | Conv 32 3×3 ReLU → MaxPool 2 → Conv 64 3×3 ReLU → MaxPool 2 → Flatten 1600 → Dense 256 ReLU → Dropout 0.4 → Dense 128 ReLU → Dropout 0.2 → Dense 10 softmax |
| Trainable params | 203,530 | 462,858 |
| Loss / optimizer | sparse CE / Adam (lr 1e-3 default) | sparse CE / Adam (lr 1e-3 default) |
| Batch / epochs | 64 / 5 | 32 / 50 (no early stopping) |
| Preprocessing | `x / 255.0`, add channel dim | same |
| Seed | `tf.random.set_seed(1234)` (not fully deterministic) | same |
| **Train acc (measured)** | **90.31 %** | **99.27 %** |
| **Test acc (measured)** | **93.10 %** (690 errors) | **95.12 %** (488 errors) |
| Test acc, de-duplicated 9,487 | 92.82 % | 94.91 % |
| Macro-F1 (test) | 0.931 | 0.951 |
| Weakest class | I (F1 0.897; 53 I→J errors) | I (F1 0.926; 29 I→J, 26 E→F, 22 G→C) |

The 99.3 % train vs 95.1 % test gap on the CNN means it overfits after 50 epochs. The
report says the opposite ("much better at generalizing").

### 2.4 Part 2 (tabular, not computer vision)

- `draft.csv`: 671 players, `position` (one-hot) + 11 numeric combine stats, label `drafted` (413 yes / 258 no).
- 85/15 split (`random_state=42`) → 570 train / 101 test. StandardScaler fit on train only (correct).
- MLP: Dense 256 ReLU L2(0.01) → BN → Dropout 0.3 → Dense 128 ReLU L2(0.01) → BN → Dropout 0.2 → Dense 1 sigmoid.
  Adam lr 5e-4, batch 16, ≤100 epochs, balanced class weights, EarlyStopping(val_accuracy, patience 10, restore best).
- **Leakage:** `validation_data=(X_test, y_test)`, so early stopping picks its checkpoint using the test set.
- **Missing values are encoded as 0.** Examples: 355/671 `threecone`, 203 `forty`, 40 `draftage`. These go into the network as real measurements.
- **No seed.** Three audit reruns gave test acc 77.2 %, 74.3 %, 77.2 %. "Not drafted" recall ranged from 0.37 to 0.66.
  Always predicting "drafted" already scores **65.3 %**. With 101 test rows, ±1 row is ±1 %.

### 2.5 Report vs. implementation

| Report claim | Code / artifact evidence | Verdict |
|---|---|---|
| MLP 90.3 % train / 93.1 % test | measured 90.31 % / 93.10 % | ✅ matches |
| CNN "about 95 %" test | measured 95.12 % | ✅ matches |
| CNN "much better at generalizing" | 99.27 % train vs 95.12 % test | ⚠️ overstated: CNN overfits more than MLP |
| MLP misclassifies test idx 148, 446, 1201 | shipped MLP gets **1201 right** (55 % conf.) | ⚠️ artifact regenerated after analysis |
| CNN confidence on idx 148 = 93.15 % | shipped CNN gives **99.80 %** | ⚠️ CNN retrained (uncommitted) after report |
| Images are "handwriting" / "digits" | notMNIST = **font-rendered letters** A–J | ❌ incorrect framing |
| Part 2 test set is "a clean portion for an unbiased check" | test set drives early stopping | ❌ leakage |
| Part 2 76 % test / 81 % train / F1(drafted) 0.81 | reruns: 74–77 % / 74–81 % / 0.83 | ≈ consistent, but unseeded and noisy |
| Changes to predict scripts (`pred_confidence` var, hard-coded model path) | code uses `prediction` and `sys.argv[2]` | ⚠️ describes a different version |

### 2.6 Authorship

The first commit (`f33c062`) shows these files came from the course template (header: *Original Author:
Jonathan Hudson, CPSC 433 F24*): `predict.py`, `predict_test.py`, `interactive.py`,
`grabimage.py`, and a starter training script (a 1-layer softmax model trained with SGD for 1 epoch).
The **uncommitted** working-tree edits replace that header with the student's name and UCID.
**Restore the original attribution before publishing.** Code you didn't write must not be presented as yours in a portfolio.
Also remove the student ID from public files.

Your own work: both model designs and their training configs, `find_misclassified.py`,
`compare_partial_complete.py`, all of Part 2, and the written analysis.

---

## 3. Current Skills vs Target Skills

| Skill | Current Status | Evidence | Target |
|---|---|---|---|
| Neural networks / MLP | IMPLEMENTED | `notMNIST-Partial.py`, `CFLModel.py` | Keep (port as baseline) |
| CNN, convolution, pooling | IMPLEMENTED | `notMNIST-Complete.py` (Conv2D, MaxPooling2D) | Improve |
| Activations (ReLU/softmax/sigmoid) | IMPLEMENTED | all three model files | Keep |
| Dropout | IMPLEMENTED | CNN (0.4/0.2), Part 2 (0.3/0.2) | Keep |
| Batch normalization | PARTIALLY IMPLEMENTED | Part 2 only (tabular) | Add to improved CNN |
| L2 regularization | PARTIALLY IMPLEMENTED | Part 2 only | Add as AdamW weight decay |
| Early stopping | PARTIALLY IMPLEMENTED | Part 2, but monitors the **test** set | Fix: monitor validation set |
| Class weighting | IMPLEMENTED | Part 2 `compute_class_weight` | Not needed (notMNIST balanced) — explain why |
| Hyperparameter tuning | PARTIALLY IMPLEMENTED | manual, test-set-driven, undocumented | Validation-driven, recorded configs |
| Train/val/test discipline | NOT IMPLEMENTED | no validation split anywhere | Add |
| Model evaluation | PARTIALLY IMPLEMENTED | accuracy only (Part 1) | Full metric suite + CIs |
| Confusion matrix | PARTIALLY IMPLEMENTED | Part 2 printed only; none for images | Add (plotted) |
| Precision / recall / F1 | PARTIALLY IMPLEMENTED | Part 2 `classification_report` only | Add per-class + macro |
| Error analysis | PARTIALLY IMPLEMENTED | list of error indices; 3 hand-picked examples | Systematic analysis |
| Prediction confidence | PARTIALLY IMPLEMENTED | softmax bars in predict scripts; not measured | Add NLL + ECE (calibration) |
| TensorFlow / Keras | IMPLEMENTED | all models | Keep as `legacy/`, evaluated on new split |
| PyTorch | NOT IMPLEMENTED | — | Add (primary framework) |
| Transfer learning | NOT IMPLEMENTED | — | Add (ResNet-18, ImageNet) |
| Fine-tuning | NOT IMPLEMENTED | — | Add (frozen → full, discriminative LR) |
| GPU training | NOT IMPLEMENTED | TF on macOS, no Metal plugin → CPU | Add (Apple MPS / CUDA auto-detect) |
| Inference | IMPLEMENTED (template) | `predict.py`, `interactive.py` (Tk) | Reusable `Predictor` + CLI |
| Model serving / API | NOT IMPLEMENTED | — | Phase 2: FastAPI |
| Docker | IMPLEMENTED | `Dockerfile`, `.dockerignore` | CPU-only image serving the FastAPI app |
| Experiment tracking | NOT IMPLEMENTED | commit messages only | Lightweight run dirs (config + metrics JSON); MLflow/W&B = future |
| Testing | NOT IMPLEMENTED | — | pytest for data, models, metrics, inference |
| Reproducibility | PARTIALLY IMPLEMENTED | TF seed set; no deps pinned; artifacts drift from report | Pinned deps, seeded splits, recorded configs |

---

## 4. What to preserve

| Existing piece | Decision | Why |
|---|---|---|
| `notMNIST.npz` | **Keep**, move to `data/` | It's the dataset. It's already in git history, so moving it adds no size. |
| `.keras` models + training scripts | **Keep as-is** in `legacy/keras_part1/` | They're the original baseline. They get re-evaluated on the new split so the comparison table includes the coursework numbers. |
| MLP & CNN architectures | **Port** to PyTorch, exactly (param counts must match: 203,530 / 462,858) | One framework, one data pipeline, one harness keeps the comparison apples-to-apples. |
| Template scripts (`predict*.py`, `interactive.py`, `grabimage.py`) | **Keep in legacy**, restore attribution; **replace** with `notmnist.inference` | The per-script argument parsing and preprocessing are duplicated three times and need TF. |
| Inverting drawn images (`1 - img`), softmax bar plot | **Reuse the idea** in the new preprocessing / error-analysis plots | It's correct domain knowledge: drawn input is dark-on-light, training data is light-on-dark. |
| `find_misclassified.py`, `compare_partial_complete.py` | **Replace** with `error_analysis.py`; keep tracking idx 148/446/1201 | Continuity with the original analysis, done systematically. |
| Part 2 (NFL draft) | **Archive** in `legacy/nfl_draft_part2/` with a README listing its known issues | Tabular, not CV. Its *techniques* (BN, weight decay, early stopping) carry into the improved CNN. |
| `433A1Files/` + zip | **Remove from the tree** (they stay in git history) | Duplicate submission bundle. |
| `env/` | **Ignore**; replace with `.venv` + `pyproject.toml` | Unpinned and 1.7 GB. |

---

## 5. Target architecture

### 5.1 Model ladder

Every model takes the **same input contract**: a float tensor `(B, 1, 28, 28)` in `[0, 1]`, with a
white glyph on a black background. Any model-specific preprocessing, such as resizing to 64×64,
3-channel replication or ImageNet normalisation for ResNet, lives **inside the model's `forward`**.
That way training, evaluation, the CLI and the API share one preprocessing function.

| Preset | Architecture | Training recipe | Question it answers |
|---|---|---|---|
| `mlp_baseline` | exact port of Keras MLP | Adam 1e-3, bs 64, 5 ep, last epoch | Reproduce coursework baseline |
| `cnn_baseline` | exact port of Keras CNN | Adam 1e-3, bs 32, 50 ep, last epoch | Reproduce coursework CNN; does spatial inductive bias help? |
| `cnn_improved` | 3 × [Conv-BN-ReLU ×2 → MaxPool → Dropout 0.1] (32/64/128 ch, padding 1) → GAP → Dropout 0.3 → Linear 10 | AdamW 1e-3, wd 5e-4, bs 128, cosine LR, ≤30 ep, early stop on val loss (patience 5), best checkpoint | Does a modern recipe (BN, GAP, weight decay, val-based stopping) close the overfitting gap with fewer params? |
| `resnet18_probe` | torchvision ResNet-18 (ImageNet weights), new 10-way head, input upsampled to 64×64 ×3 ch | backbone frozen (BN in eval mode), head only, Adam 1e-3, bs 128, 5 ep | How good are frozen ImageNet features on glyphs? |
| `resnet18_finetune` | same | 1 ep head-only, then unfreeze all; AdamW, backbone lr 1e-4 / head lr 1e-3, cosine, ≤10 ep, early stop (patience 3) | Does full fine-tuning beat a from-scratch CNN? |
| `resnet18_scratch` *(ablation, cuttable)* | same, random init | same as finetune, no freeze phase | Is any gain from **pretraining** or just from capacity? |

**Why ResNet-18 at 64×64?** At 28×28, layer4 output is 1×1, which wastes the backbone. 64×64
gives 2×2 and keeps an epoch around a minute on Apple MPS. Bigger backbones (ResNet-50, ViT)
aren't justified for 28×28 glyphs. The point is to show transfer learning *done properly*,
including the from-scratch control.

**Honest-outcome note:** the improved CNN may match or beat fine-tuned ResNet-18 here. ImageNet
features aren't obviously suited to tiny binary-ish glyphs. Either result is acceptable. Report
what you measure.

### 5.2 Data pipeline

```
notMNIST.npz ──► verify SHA-256 ──► hash every image
   train (60k) ──► drop exact duplicates (keep first) ──► stratified 90/10 split (seed 42) ──► train / val
   test  (10k) ──► flag images whose hash appears in train
                     ├── test_clean    (≈9,487)  ← PRIMARY metric
                     └── test_official (10,000)  ← for comparison with coursework numbers
```

- Scaling: `uint8 / 255 → float32`, no mean/std normalisation for the from-scratch models
  (this matches the original). ResNet applies ImageNet normalisation internally.
- No augmentation in Phase 1, so architecture comparisons aren't confounded. Augmentation is
  a Phase 3 experiment.
- All data fits in memory (≈190 MB as float32), so loading uses `TensorDataset`, `num_workers=0`
  and a seeded generator.
- Class weighting isn't used because classes are balanced to within 0.1 %. The guide should say so explicitly.

### 5.3 Training pipeline

`python -m notmnist.train --preset <name> --seed 42` creates `runs/<preset>_s<seed>/` containing:
- `config.json`: full resolved config, device, library versions, data SHA-256, split counts
- `history.csv`: per epoch: train/val loss & accuracy, LR, seconds
- `best.pt`: `{"model_name", "state_dict", "config", "class_names", "val_acc", "epoch"}`

This run directory is the experiment tracking. It's plain files and needs no extra dependency.

### 5.4 Evaluation strategy

`python -m notmnist.evaluate runs/<run>` writes `metrics.json` and `test_probs.npy`:
- accuracy with a 95 % Wilson interval, on **test_clean** (primary) and **test_official**
- macro precision / recall / F1 and per-class P/R/F1
- confusion matrix (10×10)
- NLL (log loss) and **ECE** (15 equal-width bins): is the softmax confidence trustworthy?
- trainable parameter count

`python -m notmnist.compare` builds `reports/results.md`:
- one row per run, including the **legacy Keras models** evaluated on the same test_clean split
- **McNemar's exact test** between adjacent models on the ladder (on 9.5k images, a 0.3 % difference
  may or may not be real, so the test says which)

`python -m notmnist.error_analysis` builds `reports/error_analysis.md` plus figures:
confusion-matrix heatmap, top confused pairs, a grid of the 25 most-confident errors (a label-noise
audit), errors fixed or introduced between models, and the fate of coursework indices 148/446/1201.

The test set is touched **only** by `evaluate`. Model selection, early stopping and every
hyperparameter decision use the validation split.

### 5.5 Inference

```python
from notmnist.inference import Predictor
p = Predictor.from_checkpoint("models/notmnist_cnn_improved.pt")
p.predict_image(PIL.Image.open("letter.png"), top_k=3)  # [{"label": "C", "prob": 0.97}, ...]
```
`preprocess_image` handles any size and RGB/RGBA/L input. It does not invert by default: input must
match the training polarity (light glyph on dark background), and dark-on-light images need
`invert=True` (CLI `--invert`, API `?invert=true`). An earlier mean > 0.5 auto-invert heuristic was
removed because it inverted 29.87% of test_clean (bold, filled glyphs) and dropped end-to-end
accuracy to 82.63%; with no inversion the predictor matches `notmnist.evaluate` (96.72%).
The CLI is `python -m notmnist.predict letter.png --top-k 3`, which prints JSON.

---

## 6. Target project structure

```
433A1/
├── README.md                  results table, quickstart, figures
├── PROJECT_GUIDE.md           this file
├── CLAUDE.md                  rules for AI-assisted development
├── pyproject.toml             deps + pytest config
├── .gitignore                 runs/, .venv/, env/, __pycache__, .DS_Store
├── data/notMNIST.npz
├── src/notmnist/
│   ├── __init__.py            CLASS_NAMES
│   ├── utils.py               seed_everything, get_device
│   ├── data.py                load, dedup, split, loaders
│   ├── models.py              4 architectures + build_model registry
│   ├── config.py              TrainConfig + PRESETS
│   ├── train.py               loop, early stopping, CLI
│   ├── metrics.py             pure-numpy/sklearn metric functions
│   ├── evaluate.py            run → metrics.json / test_probs.npy
│   ├── compare.py             results.md + McNemar
│   ├── error_analysis.py      figures + error_analysis.md
│   ├── inference.py           preprocess_image, Predictor
│   └── predict.py             CLI
├── legacy/
│   ├── README.md              what the coursework was, attribution, audited numbers
│   ├── evaluate_keras.py      runs in legacy env → runs/legacy_keras_*/
│   ├── keras_part1/           original PART 1 (unchanged code + .keras)
│   └── nfl_draft_part2/       original PART 2 + CSVs
├── models/                    the one checkpoint shipped for the demo
├── reports/                   results.md, error_analysis.md, figures/*.png
├── tests/
├── api/app.py                 (Phase 2)
└── Dockerfile                 (Phase 2)
```

---

## 7. Scope

### Phase 1: must have (≈9.5 h)
✅ All Phase 1 items below are done (2026-10-07): see `README.md` and `reports/`.

Repo hygiene & legacy preservation · leak-free data pipeline · PyTorch MLP & CNN reproductions ·
improved CNN · ResNet-18 linear probe + fine-tune (+ scratch ablation if time) · seeded,
config-recorded training · evaluation suite with CIs and calibration · legacy Keras models on
the same split · comparison with McNemar · error analysis · `Predictor` + CLI · README with
**measured** results · pytest for data, models, metrics and inference.

### Phase 2: if time remains (≈2 h, in priority order)
1. FastAPI `POST /predict`, `GET /health` + TestClient tests (45 min)
2. Dockerfile (CPU torch) + `docker run` smoke test (30 min)
3. 3-seed runs of `cnn_improved` and `resnet18_finetune`, reported as mean ± std (compute time only)
4. CPU latency/throughput benchmark column in `results.md` (20 min)
5. Training-curve figure per model (15 min)

### Phase 3: future roadmap (not part of the one-day build)
- **Augmentation study** (affine, elastic), measured against the Phase 1 numbers
- **Temperature scaling** on val → before/after ECE
- **Label-noise handling**: confident learning to flag/clean mislabeled training images
- **Grad-CAM** on the confusable pairs (I/J, E/F, G/C)
- **Domain shift**: evaluate on hand-drawn letters (the original Tk canvas), since the model was trained on fonts
- **ONNX export** + onnxruntime latency comparison
- GitHub Actions CI running pytest; MLflow if the run count grows
- Hyperparameter search (Optuna) on `cnn_improved`
- EMNIST-letters (26 classes, handwritten) as a second dataset

---

## 8. One-day implementation schedule

Detailed, test-first steps are in `docs/superpowers/plans/2026-10-07-notmnist-portfolio.md`.

| # | Task | Time | Depends on | Output | Verified by |
|---|---|---|---|---|---|
| 0 | Repo hygiene, legacy move, attribution, env, package skeleton | 0.75 h | — | `legacy/`, `pyproject.toml`, `.venv`, `utils.py` | `pytest` collects; `python -c "import notmnist"` |
| 1 | Data pipeline (dedup, split, loaders) | 1 h | 0 | `data.py` | split tests: no train/val hash overlap; test_clean has no train hash; determinism |
| 2 | Model zoo | 1 h | 0 | `models.py` | param counts 203,530 / 462,858; all output `(B,10)` |
| 3 | Training loop + presets + run dirs | 1.25 h | 1, 2 | `config.py`, `train.py` | 1-epoch smoke run on 2k-sample subset writes all artifacts |
| 4 | Metrics + evaluate CLI | 1 h | 3 | `metrics.py`, `evaluate.py` | metric unit tests on hand-computed cases |
| 5 | Legacy Keras on the new split | 0.5 h | 1, 4 | `legacy/evaluate_keras.py` | test_official acc = 93.10 % / 95.12 % (matches audit) |
| 6 | Run the experiments | 1.5 h (mostly compute) | 3, 4 | `runs/*` | every run has `metrics.json` |
| 7 | Comparison + McNemar | 0.75 h | 5, 6 | `compare.py`, `reports/results.md` | McNemar unit test; table rows = runs |
| 8 | Error analysis | 1 h | 6 | `error_analysis.py`, `reports/` | figures exist; written observations come from actual images |
| 9 | Predictor + CLI | 0.75 h | 3 | `inference.py`, `predict.py`, `models/` | PNG round-trip test; inverted-polarity test |
| 10 | README + docs | 0.5 h | 7, 8, 9 | `README.md` | every number traceable to a `metrics.json` |
| | **Phase 1 total** | **≈10 h** | | | |
| 11 | FastAPI | 0.75 h | 9 | `api/app.py` | TestClient tests |
| 12 | Docker | 0.5 h | 11 | `Dockerfile` | `curl` against running container |

Run Task 6's training jobs in the background while working on Tasks 7–9.

---

## 9. Verification strategy

- **Unit tests** (`pytest -q`) for data split invariants, model shapes and param counts, metric correctness,
  checkpoint round-trip, preprocessing polarity, and API contract (Phase 2).
- **Reproduction check:** PyTorch `mlp_baseline` / `cnn_baseline` should land near the Keras numbers
  (≈93 % / ≈95 % on test_official). A large gap means the port is wrong. Look at the port before reading anything into it.
- **Legacy cross-check:** `evaluate_keras.py` must reproduce the audited 93.10 % / 95.12 %.
- **Traceability:** every number in README/results comes from a `metrics.json` produced by a command
  listed next to it.
- **No silent success:** a training run that crashes or diverges is reported as such.

---

## 10. Resume-relevant skills (after Phase 1)

PyTorch · CNN design (BatchNorm, global average pooling, dropout, weight decay) · transfer
learning & fine-tuning (frozen-BN linear probe, discriminative learning rates, pretrained-vs-scratch
ablation) · GPU/MPS training · data-leakage detection and leak-free splitting · evaluation methodology
(validation-based selection, CIs, McNemar, calibration/ECE) · error analysis · reproducible
experiment tracking · inference packaging · pytest. Phase 2 adds FastAPI and Docker.

---

## 11. Resume positioning

**Right now (honest, coursework):**
> **notMNIST Letter Classification**, *Course project, CPSC 433, University of Calgary* · Python, TensorFlow/Keras
> - Built and compared an MLP and a CNN for 10-class glyph recognition on notMNIST (70k images); the CNN raised test accuracy from 93.1 % to 95.1 %.
> - Built a regularised tabular classifier (L2, batch norm, dropout, class weighting, early stopping) predicting NFL draft outcomes from combine data.

**After the one-day expansion (measured values filled in from `reports/results.md` and `reports/error_analysis.md`; the FastAPI and Docker parts are done):**
> **notMNIST Letter Recognition: from MLP to Transfer Learning** · PyTorch, torchvision, scikit-learn, pytest *(+ FastAPI, Docker)*
> - Found and removed train/test leakage (5.1 % of test images duplicated in training). Rebuilt evaluation on a de-duplicated, validation-based split with 95 % CIs, McNemar significance tests and calibration (ECE).
> - Benchmarked 5 architectures on a single pipeline (MLP, CNN, BatchNorm CNN, and ResNet-18 as both a frozen probe and a full fine-tune, plus a from-scratch ablation). Reached 96.72 % test_clean accuracy (macro-F1 0.9668) vs. a 92.82 % Keras MLP baseline, using 1.6× fewer parameters than the original CNN (288,618 vs 462,858); pretrained and from-scratch ResNet-18 were not significantly different (McNemar p = 0.5044) and both scored below the CNN.
> - Error analysis showed the 10 most frequent confusions (led by J→I with 29 errors, then H→A with 15) account for 125 of 311 remaining errors (40.2 %), and a manual audit of the 25 most-confident errors found 4 likely mislabels and 9 unreadable or decorative glyphs.
> - Packaged a reproducible inference path (CLI *(and FastAPI service in Docker)*) with pytest coverage of data splits, metrics and preprocessing.

The "fewer parameters" claim is supported: 462,858 / 288,618 = 1.60×. Drop the italicised FastAPI/Docker text until Phase 2 is built.
