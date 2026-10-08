# notMNIST Portfolio Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the CPSC 433 Keras notMNIST assignment into a reproducible PyTorch CV project with a leak-free split, a 5-model ladder (MLP → CNN → improved CNN → ResNet-18 probe/fine-tune), rigorous evaluation, error analysis and a CLI predictor.

**Architecture:** An installable `src/notmnist` package. The data pipeline de-duplicates and splits the data once, deterministically. Every model takes `(B,1,28,28)` in [0,1], and model-specific preprocessing lives inside `forward`. Training writes self-describing run directories. Evaluation writes `test_probs.npy` + `metrics.json`, and comparison and error analysis read only those files, so legacy Keras models plug in through the same interface.

**Tech Stack:** Python 3.12 (uv venv), PyTorch ≥2.4 + torchvision (MPS/CUDA/CPU), NumPy, scikit-learn, matplotlib, Pillow, pytest. Phase 2: FastAPI, uvicorn, python-multipart, httpx, Docker.

**Spec:** `PROJECT_GUIDE.md` (§5 target architecture, §7 scope). Rules: `CLAUDE.md`.

## Global Constraints

- Class names: `CLASS_NAMES = ["A","B","C","D","E","F","G","H","I","J"]`, label index = position.
- Data file: `data/notMNIST.npz`, keys `x_train (60000,28,28) uint8`, `y_train`, `x_test (10000,28,28)`, `y_test`.
- Default seed `42`. Validation fraction `0.1`, stratified, carved from the **de-duplicated** train set.
- Model input contract: `torch.float32 (B,1,28,28)` in [0,1], light glyph on dark background. Output: logits `(B,10)`.
- Primary test metric on `test_clean` (test images whose bytes do not appear in train); secondary on `test_official` (all 10,000).
- The test set is read only in `notmnist/evaluate.py` and `legacy/evaluate_keras.py`.
- Run dir: `runs/<preset>_s<seed>/` with `config.json`, `history.csv`, `best.pt`; evaluation adds `metrics.json`, `test_probs.npy` (float32, shape `(10000,10)`, official test order).
- Checkpoint dict keys: `model_name, state_dict, config, class_names, epoch, val_acc`.
- No new dependencies beyond the Tech Stack line. `legacy/` code is moved, not modified (attribution headers excepted).
- Every reported number comes from a `metrics.json`. If something wasn't run, write "not run".

## Review Focus

1. **Wrong-polarity or odd-format user images.** A dark-on-light, RGB/RGBA, non-28×28 PNG should be auto-inverted, converted and resized, then get the same prediction as the original sample. A missing inversion gives confident garbage. *(Task 9: `test_inverted_rgb_large_image_matches`)*
2. **Duplicate leakage across train/val.** Hashes must not overlap between train and val, and test_clean must contain no train hash. *(Task 1: `test_no_hash_overlap`)*
3. **Train-mode layers at inference.** Dropout/BN active during evaluation or prediction gives non-deterministic, worse output. Repeated predictions must be identical. *(Task 9: `test_predict_is_deterministic`; Task 4 uses `model.eval()`)*
4. **Checkpoint reload without network or presets.** Loading a ResNet checkpoint must not download ImageNet weights and must reproduce the logits exactly. *(Task 3: `test_checkpoint_roundtrip`; Task 2: `build_model(..., pretrained=False)`)*
5. **BN running-stat drift in the frozen phase.** Linear-probe training must not change backbone BN buffers. *(Task 3: `test_frozen_backbone_bn_unchanged`)*

---

### Task 0: Repo hygiene, legacy preservation, package skeleton

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `src/notmnist/__init__.py`, `src/notmnist/utils.py`, `legacy/README.md`, `tests/test_utils.py`
- Move (git mv): `PART 1/*` → `legacy/keras_part1/`, `PART 2/*` + `draft_*.csv` → `legacy/nfl_draft_part2/`, `PART 1/notMNIST.npz` → `data/notMNIST.npz`
- Delete from tree: `433A1Files/`, `433A1Files.zip`, tracked `.DS_Store`

**Interfaces:**
- Produces: `notmnist.CLASS_NAMES: list[str]`; `notmnist.utils.seed_everything(seed: int) -> torch.Generator`; `notmnist.utils.get_device(prefer: str | None = None) -> torch.device` (order mps → cuda → cpu unless `prefer`); `notmnist.utils.REPO_ROOT: Path`; `notmnist.utils.DATA_PATH = REPO_ROOT / "data/notMNIST.npz"`.

- [ ] **Step 1: Snapshot the current state.** Commit the existing uncommitted working tree first, so the coursework's final state is in history: `git add -A "PART 1" "PART 2" && git commit -m "chore: snapshot final coursework state"`. **First ask the user** whether to restore the original template attribution headers (`Original Author: Jonathan Hudson / CPSC 433 F24`) in `predict.py`, `predict_test.py`, `interactive.py` and `grabimage.py`, and whether to drop the UCID line from all headers. Apply the answer in this commit.
- [ ] **Step 2: Move files** with `git mv` as listed. Add `.gitignore`: `.venv/ env/ runs/ __pycache__/ *.pyc .DS_Store .pytest_cache/`. Remove `433A1Files/` and the zip.
- [ ] **Step 3: Write `legacy/README.md`.** Copy facts from PROJECT_GUIDE §2.3–2.6 (architectures, audited numbers, known issues, attribution). Note that the scripts expect to run from their own directory and that `notMNIST.npz` now lives in `data/`.
- [ ] **Step 4: `pyproject.toml`** (setuptools, `src` layout, package `notmnist`). Deps: `torch>=2.4`, `torchvision>=0.19`, `numpy`, `scikit-learn`, `matplotlib`, `pillow`. Extras: `dev = [pytest]`, `api = [fastapi, uvicorn, python-multipart, httpx]`. Add `[tool.pytest.ini_options] testpaths=["tests"]`. Run `uv venv --python 3.12 .venv && uv pip install -e ".[dev]"`.
- [ ] **Step 5: Write failing tests** in `tests/test_utils.py`:
```python
def test_seed_everything_reproducible():
    seed_everything(0); a = torch.rand(3)
    seed_everything(0); b = torch.rand(3)
    assert torch.equal(a, b)

def test_get_device_prefer_cpu():
    assert get_device("cpu").type == "cpu"

def test_data_file_exists():
    assert DATA_PATH.exists()
```
- [ ] **Step 6:** `.venv/bin/pytest tests/test_utils.py -v` → FAIL (import error).
- [ ] **Step 7: Implement `utils.py`.** `seed_everything` seeds `random`, `numpy`, `torch` and calls `torch.use_deterministic_algorithms(True, warn_only=True)`. It returns a seeded `torch.Generator` for DataLoaders. Put `CLASS_NAMES` in `__init__.py`.
- [ ] **Step 8:** `.venv/bin/pytest -q` → PASS. Also `.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"` → `True` on this M4.
- [ ] **Step 9: Commit** `chore: restructure into legacy/ + installable notmnist package`.

---

### Task 1: Leak-free data pipeline

**Files:** Create `src/notmnist/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `DATA_PATH`, `seed_everything`.
- Produces:
  - `load_raw(path: Path = DATA_PATH) -> dict[str, np.ndarray]`
  - `image_hashes(x: np.ndarray) -> np.ndarray` (dtype object/str, MD5 hex of each image's bytes)
  - `@dataclass Splits: x_train, y_train, x_val, y_val, x_test, y_test (uint8 arrays); test_clean_mask: np.ndarray[bool]; info: dict`
  - `make_splits(seed: int = 42, val_frac: float = 0.1, path: Path = DATA_PATH) -> Splits`. `info` keys: `sha256, n_train_raw, n_train_dedup, n_train, n_val, n_test, n_test_clean, n_train_dup_conflicting_labels`
  - `to_tensor(x: np.ndarray) -> torch.Tensor` (uint8 `(N,28,28)` → float32 `(N,1,28,28)` /255)
  - `make_loader(x, y, batch_size: int, shuffle: bool, generator: torch.Generator | None = None) -> DataLoader` (TensorDataset, `num_workers=0`)
  - **Constraint:** `data.py` imports torch *inside* `to_tensor`/`make_loader` only, so the legacy TF env (no torch) can import `make_splits`.

- [ ] **Step 1: Write failing tests** (use one module-scoped fixture `splits = make_splits(42)`):
```python
def test_counts(splits):
    i = splits.info
    assert i["n_train_raw"] == 60000 and i["n_test"] == 10000
    assert i["n_train_dedup"] == 58239                 # audited
    assert i["n_train"] + i["n_val"] == 58239
    assert i["n_test_clean"] == 10000 - 513            # audited

def test_no_hash_overlap(splits):
    tr, va = set(image_hashes(splits.x_train)), set(image_hashes(splits.x_val))
    te_clean = set(image_hashes(splits.x_test[splits.test_clean_mask]))
    assert not tr & va
    assert not (tr | va) & te_clean

def test_val_is_stratified(splits):
    counts = np.bincount(splits.y_val, minlength=10)
    assert counts.max() - counts.min() <= 20

def test_deterministic():
    a, b = make_splits(42), make_splits(42)
    assert np.array_equal(a.y_val, b.y_val) and np.array_equal(a.x_train[:50], b.x_train[:50])

def test_to_tensor_range_and_shape(splits):
    t = to_tensor(splits.x_val[:4])
    assert t.shape == (4, 1, 28, 28) and t.dtype == torch.float32 and 0 <= t.min() and t.max() <= 1
```
- [ ] **Step 2:** `.venv/bin/pytest tests/test_data.py -v` → FAIL.
- [ ] **Step 3: Implement `data.py`.** Dedup keeps the first occurrence (`np.unique(hashes, return_index=True)`, then sort the indices to preserve order). Count duplicate groups whose labels disagree and put the count in `info["n_train_dup_conflicting_labels"]`. Stratify with `sklearn.model_selection.train_test_split(stratify=y, random_state=seed)`. `test_clean_mask = ~np.isin(test_hashes, train_raw_hashes)`, using the *raw* train hashes so val duplicates also count.
- [ ] **Step 4:** tests → PASS. Print `make_splits().info` and record the conflicting-label count. It's a fact for the error analysis.
- [ ] **Step 5: Commit** `feat(data): leak-free dedup + stratified val split`.

---

### Task 2: Model zoo

**Files:** Create `src/notmnist/models.py`, `tests/test_models.py`

**Interfaces:**
- Produces:
  - `MLPBaseline(nn.Module)`, `CNNBaseline(nn.Module)`, `CNNImproved(nn.Module)`, `ResNet18Transfer(nn.Module)` with attributes `.backbone` (everything but the head) and `.head: nn.Linear(512, 10)`
  - `MODEL_NAMES = ("mlp_baseline", "cnn_baseline", "cnn_improved", "resnet18")`
  - `build_model(name: str, pretrained: bool = False) -> nn.Module` (`pretrained` only affects resnet18; `False` must never touch the network)
  - `count_params(model: nn.Module) -> int` (trainable only)

Architectures (PROJECT_GUIDE §5.1):
- `MLPBaseline`: Flatten → Linear(784,256) → ReLU → Linear(256,10).
- `CNNBaseline`: Conv2d(1,32,3) ReLU MaxPool2 → Conv2d(32,64,3) ReLU MaxPool2 → Flatten → Linear(1600,256) ReLU Dropout(0.4) → Linear(256,128) ReLU Dropout(0.2) → Linear(128,10). (Keras Dropout rate = PyTorch `p`.)
- `CNNImproved`: 3 blocks with channels 32/64/128; each block is [Conv3×3 pad1 → BN → ReLU] ×2 → MaxPool2 → Dropout(0.1). Then AdaptiveAvgPool2d(1) → Flatten → Dropout(0.3) → Linear(128,10).
- `ResNet18Transfer`: `forward` interpolates to 64×64 (bilinear, `align_corners=False`), repeats to 3 channels, normalises with ImageNet mean `(0.485,0.456,0.406)` / std `(0.229,0.224,0.225)` (registered buffers), then runs torchvision `resnet18(weights=IMAGENET1K_V1 if pretrained else None)` with `fc` replaced by `Linear(512,10)`.

- [ ] **Step 1: Write failing tests**:
```python
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_output_shape(name):
    m = build_model(name).eval()
    assert m(torch.rand(2, 1, 28, 28)).shape == (2, 10)

def test_ports_match_keras_param_counts():
    assert count_params(build_model("mlp_baseline")) == 203_530
    assert count_params(build_model("cnn_baseline")) == 462_858

def test_improved_cnn_is_smaller_than_baseline():
    assert count_params(build_model("cnn_improved")) < 462_858

def test_resnet_has_head_and_backbone():
    m = build_model("resnet18")
    assert isinstance(m.head, nn.Linear) and m.head.out_features == 10
```
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** run → PASS. Print `count_params` for all four models and record the numbers.
- [ ] **Step 5: Commit** `feat(models): MLP/CNN Keras ports, improved CNN, ResNet-18 transfer`.

---

### Task 3: Training loop, presets, run directories

**Files:** Create `src/notmnist/config.py`, `src/notmnist/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `make_splits`, `make_loader`, `to_tensor`, `build_model`, `seed_everything`, `get_device`.
- Produces:
  - `@dataclass TrainConfig: preset: str; model_name: str; pretrained: bool = False; optimizer: str = "adam"  # "adam"|"adamw"; lr: float = 1e-3; backbone_lr: float | None = None; weight_decay: float = 0.0; batch_size: int = 64; epochs: int = 5; scheduler: str = "none"  # "none"|"cosine"; early_stopping_patience: int | None = None; freeze_backbone_epochs: int = 0; seed: int = 42; subset: int | None = None`
  - `PRESETS: dict[str, TrainConfig]`. Values from PROJECT_GUIDE §5.1:
    - `mlp_baseline`: adam 1e-3, bs 64, 5 ep
    - `cnn_baseline`: adam 1e-3, bs 32, 50 ep
    - `cnn_improved`: adamw 1e-3, wd 5e-4, bs 128, 30 ep, cosine, patience 5
    - `resnet18_probe`: pretrained, adam 1e-3, bs 128, 5 ep, freeze_backbone_epochs 5
    - `resnet18_finetune`: pretrained, adamw lr 1e-3, backbone_lr 1e-4, wd 1e-4, bs 128, 10 ep, cosine, patience 3, freeze_backbone_epochs 1
    - `resnet18_scratch`: same as finetune, but `pretrained=False`, freeze 0, `backbone_lr=None`
  - `train(cfg: TrainConfig, out_dir: Path, device: torch.device | None = None) -> Path` (returns the `best.pt` path)
  - `notmnist.models.load_checkpoint(path: Path, device="cpu") -> tuple[nn.Module, dict]`: lives in **`models.py`** (added in this task) so inference never imports training code. Rebuilds with `pretrained=False`, loads state, returns `.eval()` model + checkpoint dict.
  - `set_backbone_trainable(model: ResNet18Transfer, trainable: bool) -> None`
  - CLI: `python -m notmnist.train --preset NAME [--seed N] [--subset N] [--device cpu|mps|cuda]` → `runs/<preset>_s<seed>/`

Behaviour:
- Loss: `nn.CrossEntropyLoss` on logits.
- The val split is evaluated every epoch.
- With `early_stopping_patience`: track best val loss, save `best.pt` on improvement, stop after `patience` epochs without improvement.
- Without it: save the last epoch, to match the Keras originals.
- While frozen, backbone params get `requires_grad=False` and the backbone is put in `.eval()` after every `model.train()` call.
- The optimizer has two param groups when `backbone_lr` is set.
- After the freeze phase, unfreeze and rebuild optimizer + scheduler for the remaining epochs.
- `config.json` includes `asdict(cfg)`, device, `torch.__version__`, `splits.info`, n_params.
- `subset` truncates train/val (first N after shuffling with the seed). It's for smoke tests only.

- [ ] **Step 1: Write failing tests** (CPU, `subset=512`, 1 epoch):
```python
def test_smoke_run_writes_artifacts(tmp_path):
    cfg = replace(PRESETS["cnn_improved"], epochs=1, subset=512)
    ckpt = train(cfg, tmp_path, device=torch.device("cpu"))
    for f in ["config.json", "history.csv", "best.pt"]:
        assert (tmp_path / f).exists()
    assert json.loads((tmp_path / "config.json").read_text())["config"]["preset"] == "cnn_improved"

def test_checkpoint_roundtrip(tmp_path):
    cfg = replace(PRESETS["resnet18_scratch"], epochs=1, subset=256)
    ckpt = train(cfg, tmp_path, device=torch.device("cpu"))
    m, meta = load_checkpoint(ckpt)
    x = torch.rand(3, 1, 28, 28)
    assert meta["class_names"] == CLASS_NAMES
    assert torch.allclose(m(x), m(x))      # eval mode, deterministic

def test_frozen_backbone_bn_unchanged(tmp_path):
    cfg = replace(PRESETS["resnet18_probe"], pretrained=False, epochs=1, subset=256)
    # build via train(); compare a backbone BN running_mean before/after
    ...assert torch.equal(before, after)   # use build_model + seed to get "before"
```
(For the BN test, expose `train(..., return_model=True)` *or* load `best.pt` and compare against `build_model("resnet18")` initialised with the same seed. Pick whichever needs less code and state the choice in the commit.)
- [ ] **Step 2:** run → FAIL. **Step 3:** implement `config.py` and `train.py`. **Step 4:** run → PASS.
- [ ] **Step 5: Real smoke test on MPS.** Run `python -m notmnist.train --preset cnn_baseline --subset 2000 --device mps`. It should finish, `history.csv` should show val acc rising, and seconds/epoch should be recorded. Note the time per epoch, then estimate Task 6's total compute and report it.
- [ ] **Step 6: Commit** `feat(train): presets, early stopping, freeze/unfreeze, run dirs`.

---

### Task 4: Metrics and the evaluate CLI

**Files:** Create `src/notmnist/metrics.py`, `src/notmnist/evaluate.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `load_checkpoint`, `make_splits`, `to_tensor`.
- Produces:
  - `wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]`
  - `expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float`
  - `compute_metrics(probs: np.ndarray, y: np.ndarray) -> dict` with keys `n, accuracy, accuracy_ci95, macro_precision, macro_recall, macro_f1, per_class {name: {precision, recall, f1, support}}, confusion_matrix (list[list[int]]), nll, ece`
  - `notmnist.evaluate.predict_probs(model, x_uint8: np.ndarray, device, batch_size=512) -> np.ndarray` (softmax under `torch.inference_mode()`). `metrics.py` stays numpy/sklearn-only (no torch import) so the legacy env can use it.
  - `evaluate_run(run_dir: Path, device=None) -> dict`. Writes `test_probs.npy` and `metrics.json = {"run", "model_name", "n_params", "val": {...}, "test_clean": {...}, "test_official": {...}}`
  - CLI: `python -m notmnist.evaluate RUN_DIR [RUN_DIR ...]`

- [ ] **Step 1: Write failing tests:**
```python
def test_perfect_predictions():
    y = np.array([0, 1, 2, 2]); p = np.eye(10)[y]
    m = compute_metrics(p, y)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["ece"] == pytest.approx(0.0)

def test_known_accuracy_and_confusion():
    y = np.array([0, 0, 1, 1]); pred = np.array([0, 1, 1, 1])
    m = compute_metrics(np.eye(10)[pred] * 0.9 + 0.01, y)
    assert m["accuracy"] == 0.75 and m["confusion_matrix"][0][1] == 1

def test_wilson_ci_brackets_estimate():
    lo, hi = wilson_ci(950, 1000)
    assert lo < 0.95 < hi and hi - lo < 0.03

def test_ece_overconfident():
    y = np.zeros(100, dtype=int); p = np.full((100, 10), 0.0); p[:, 0] = 1.0; p[:50, 0], p[:50, 1] = 0.0, 1.0
    assert compute_metrics(p, y)["ece"] == pytest.approx(0.5)
```
- [ ] **Step 2:** FAIL. **Step 3:** implement. Use `sklearn.metrics.precision_recall_fscore_support`, `confusion_matrix(labels=range(10))` and `log_loss(labels=range(10))`. Write ECE yourself (≈10 lines: bin by max prob, weighted |acc − conf|). Only per-class/macro metrics restrict to classes in the confusion matrix. **Step 4:** PASS.
- [ ] **Step 5:** Run `python -m notmnist.evaluate` on the Task 3 smoke run dir. `metrics.json` must have all three sections and `test_probs.npy` must have shape `(10000,10)`.
- [ ] **Step 6: Commit** `feat(eval): metrics with CI, NLL, ECE; evaluate CLI`.

---

### Task 5: Legacy Keras models on the new split

**Files:** Create `legacy/evaluate_keras.py`

**Interfaces:**
- Consumes: `metrics.compute_metrics`, `data.make_splits`. Import them by adding `src/` to `sys.path` (both are torch-free at import time per Tasks 1 and 4). Don't import the `notmnist` package `__init__` chain if it pulls in torch; keep `__init__.py` torch-free.
- Produces: `runs/legacy_keras_mlp/` and `runs/legacy_keras_cnn/`, each with `test_probs.npy` + `metrics.json` in the Task 4 schema (`model_name` = `"keras_mlp"` / `"keras_cnn"`, `n_params` from `model.count_params()`, `val` = `null` because the Keras models never saw this val split, but they *did* train on all 60k, so val would leak).

- [ ] **Step 1: Implement.** It loads both `.keras` files from `legacy/keras_part1/`, predicts on `splits.x_test / 255.0` with a channel dim, and writes the outputs.
- [ ] **Step 2: Verify against the audit.** Run `env/bin/python legacy/evaluate_keras.py`. Expect test_official accuracy **0.9310** (MLP) and **0.9512** (CNN), and test_clean **0.9282 / 0.9491**. Any other value means a bug, so stop and investigate.
- [ ] **Step 3: Commit** `feat(legacy): evaluate original Keras models on the shared test split`.

---

### Task 6: Run the experiments

**Files:** none (produces `runs/*`)

- [ ] **Step 1:** Launch these in sequence, in the background (one shell loop):
`mlp_baseline`, `cnn_baseline`, `cnn_improved`, `resnet18_probe`, `resnet18_finetune`, then `resnet18_scratch` **only if** the Task 3 timing estimate puts the total at ≤ 90 min. Use `--seed 42` for all. Then `python -m notmnist.evaluate runs/*_s42`.
- [ ] **Step 2: Sanity checks.**
  - PyTorch `mlp_baseline` / `cnn_baseline` test_official should be within ~1 pp of the Keras 93.1 % / 95.1 %. Report the actual gap. If it's larger than 2 pp, investigate the port before continuing.
  - Every run has `metrics.json`.
  - Check each `history.csv` for divergence (NaN loss, val acc near 10 %).
- [ ] **Step 3:** Report the raw numbers to the user (a table pasted straight from the JSON) before writing any narrative.

---

### Task 7: Model comparison

**Files:** Create `src/notmnist/compare.py`, `tests/test_compare.py`; output `reports/results.md`, `reports/results.json`

**Interfaces:**
- Consumes: `runs/*/metrics.json`, `runs/*/test_probs.npy`, `make_splits` (for `y_test`, `test_clean_mask`).
- Produces:
  - `mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict` with `{"b01": int, "b10": int, "p_value": float}`. The two-sided exact binomial uses `math.comb`; with 0 discordant pairs, p = 1.0.
  - `LADDER = [("legacy_keras_mlp","mlp_baseline_s42"), ("mlp_baseline_s42","cnn_baseline_s42"), ("cnn_baseline_s42","cnn_improved_s42"), ("cnn_improved_s42","resnet18_finetune_s42"), ("resnet18_probe_s42","resnet18_finetune_s42"), ("resnet18_scratch_s42","resnet18_finetune_s42")]` (skip pairs whose run is missing)
  - CLI `python -m notmnist.compare` → `reports/results.md`. The main table has columns: run, params, val acc, test_clean acc [95 % CI], test_official acc, macro-F1, NLL, ECE. A second table lists ladder pairs with b01/b10/p, all on test_clean.

- [ ] **Step 1: Write failing tests:**
```python
def test_mcnemar_identical():
    c = np.array([True, False, True])
    assert mcnemar_exact(c, c)["p_value"] == 1.0

def test_mcnemar_known_value():
    a = np.ones(10, dtype=bool); b = np.zeros(10, dtype=bool)
    r = mcnemar_exact(a, b)
    assert r["b10"] == 10 and r["b01"] == 0 and r["p_value"] == pytest.approx(2 * 0.5**10)
```
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS.
- [ ] **Step 5:** Run it and open `reports/results.md`. Spot-check two cells against their `metrics.json`.
- [ ] **Step 6: Commit** `feat(compare): results table + McNemar ladder` (include `reports/`).

---

### Task 8: Error analysis

**Files:** Create `src/notmnist/error_analysis.py`; output `reports/error_analysis.md`, `reports/figures/*.png`

**Interfaces:**
- Consumes: run `test_probs.npy`, `make_splits`, `CLASS_NAMES`.
- Produces: CLI `python -m notmnist.error_analysis [--best RUN] [--baseline RUN]`. Defaults: best = highest test_clean accuracy in `reports/results.json`; baseline = `mlp_baseline_s42`. All analysis is on test_clean, except the coursework indices, which are official-test indices.
  - `figures/confusion_<run>.png`: row-normalised heatmap
  - `figures/confident_errors_<run>.png`: 5×5 grid of the most confident errors, titled `true→pred (p)`
  - `figures/coursework_indices.png`: idx 148, 446, 1201 with each model's top-1 + confidence
  - `error_analysis.md`: top-10 confused pairs (count, % of errors); per-class F1 for baseline vs best; fixed/introduced error counts baseline→best; confidence histogram of correct vs wrong (one figure); and the coursework-index table.

- [ ] **Step 1: Implement** (plotting code, no unit tests). Data selection is a pure function `top_confusions(y, pred, k=10) -> list[tuple[int,int,int]]`. Write one test for it with a 3-sample hand case.
- [ ] **Step 2:** Run it, then **open the generated PNGs** (Read tool) and write the observations section of `error_analysis.md` from what is actually visible. Count how many of the 25 confident errors look mislabeled or unreadable, and label that count "manual judgement".
- [ ] **Step 3: Commit** `feat(analysis): error analysis report and figures`.

---

### Task 9: Inference (Predictor + CLI)

**Files:** Create `src/notmnist/inference.py`, `src/notmnist/predict.py`, `tests/test_inference.py`; copy the chosen checkpoint to `models/notmnist_<preset>.pt`

**Interfaces:**
- Consumes: `notmnist.models.load_checkpoint`, `CLASS_NAMES`. **Must not import** `train`, `data` or `evaluate`.
- Produces:
  - `preprocess_image(img: PIL.Image.Image, invert: bool | None = None) -> torch.Tensor` (`(1,1,28,28)` float in [0,1]). Steps: convert RGBA→composite on white→`L`; resize to 28×28 (LANCZOS); scale /255; when `invert is None`, invert if mean > 0.5.
  - `class Predictor`: `from_checkpoint(path, device="cpu") -> Predictor`; `predict_tensor(x: Tensor) -> np.ndarray` (probs `(B,10)`); `predict_image(img, top_k=3) -> list[dict]` with `{"label": str, "index": int, "prob": float}` sorted by prob descending; `.model_name: str`
  - CLI: `python -m notmnist.predict IMAGE [--checkpoint models/...] [--top-k 3] [--invert/--no-invert]`, which prints JSON

Deployment checkpoint choice: the smallest model whose test_clean accuracy is within its 95 % CI of the best. Likely `cnn_improved`, but decide from `results.md` and write the reason in README.

- [ ] **Step 1: Write failing tests** (use a fixture checkpoint from a 1-epoch `subset=512` CPU training run on `cnn_improved`, `scope="module"`):
```python
def test_png_roundtrip_matches_tensor(predictor, splits, tmp_path):
    x = splits.x_val[0]; Image.fromarray(x).save(tmp_path / "a.png")
    p_img = predictor.predict_image(Image.open(tmp_path / "a.png"), top_k=10)
    p_ten = predictor.predict_tensor(to_tensor(x[None]))[0]
    assert p_img[0]["index"] == int(p_ten.argmax())

def test_inverted_rgb_large_image_matches(predictor, splits):
    x = splits.x_val[0]
    big = Image.fromarray(255 - x).resize((200, 200)).convert("RGB")
    assert predictor.predict_image(big, top_k=1)[0]["index"] == predictor.predict_image(Image.fromarray(x), top_k=1)[0]["index"]

def test_predict_is_deterministic(predictor, splits):
    img = Image.fromarray(splits.x_val[1])
    assert predictor.predict_image(img) == predictor.predict_image(img)

def test_topk_sorted_and_sums(predictor, splits):
    out = predictor.predict_image(Image.fromarray(splits.x_val[2]), top_k=10)
    assert [o["prob"] for o in out] == sorted((o["prob"] for o in out), reverse=True)
    assert sum(o["prob"] for o in out) == pytest.approx(1.0, abs=1e-4)
```
(The test file may import `data` for fixtures; only `inference.py` itself is restricted.)
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS.
- [ ] **Step 5:** Copy the chosen checkpoint to `models/`. Run the CLI on `legacy/keras_part1/image.png` (the original hand-drawn sample) and on a saved test PNG. Record both outputs; the hand-drawn one is out-of-distribution, so whatever it outputs is just recorded, not judged.
- [ ] **Step 6: Commit** `feat(inference): Predictor, preprocessing, CLI, demo checkpoint`.

---

### Task 10: README and documentation

**Files:** Create `README.md`; modify `PROJECT_GUIDE.md` (§2 "current state" → add a "Results" pointer; tick Phase 1), `CLAUDE.md` (commands, if they changed)

- [ ] **Step 1:** README sections:
  - one-paragraph pitch
  - results table, copied from `reports/results.md` and labelled with the command that produced it
  - 2–3 figures
  - key findings: leakage, the ladder, the pretrained-vs-scratch result, calibration
  - quickstart: setup, train, evaluate, predict
  - project structure
  - "Origins" paragraph linking `legacy/README.md`, with attribution
  - limitations
- [ ] **Step 2:** Fill the resume bullet placeholders in PROJECT_GUIDE §11 with measured values only.
- [ ] **Step 3:** Run `.venv/bin/pytest -q`. Everything must pass; paste the summary line.
- [ ] **Step 4: Commit** `docs: README with measured results`.

---

## Phase 2 (only after Phase 1 is verified)

### Task 11: FastAPI endpoint

**Files:** Create `api/app.py`, `tests/test_api.py`. Install with `uv pip install -e ".[dev,api]"`.

**Interfaces:**
- Consumes: `Predictor`.
- Produces:
  - `app = FastAPI()`
  - `GET /health` → `{"status":"ok","model":<model_name>}`
  - `POST /predict` (multipart `file`, optional query `top_k: int = 3`, range 1–10) → `{"model": str, "predictions": [{"label","index","prob"}]}`
  - Errors: non-image or corrupt → 400; >1 MB → 413
  - The checkpoint path comes from env `NOTMNIST_CHECKPOINT` (default `models/notmnist_<preset>.pt`) and is loaded once at startup.

- [ ] **Step 1: Write failing tests** with `fastapi.testclient.TestClient`: health 200; predict on a PNG of `x_val[0]` returns 3 sorted predictions; a text file returns 400; `top_k=11` returns 422.
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS → **Step 5: Commit** `feat(api): FastAPI predict endpoint`.

### Task 12: Docker

**Files:** Create `Dockerfile`, `.dockerignore` (exclude `env/ .venv/ runs/ data/ legacy/ .git/`)

- [ ] **Step 1:** Base `python:3.12-slim`. Install CPU torch with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then `pip install ".[api]"`. Copy `src/`, `api/`, `models/`. Run `CMD uvicorn api.app:app --host 0.0.0.0 --port 8000`.
- [ ] **Step 2:** Check free disk space first (the machine has ~20 GB). Then `docker build -t notmnist-api .` and `docker run -p 8000:8000 notmnist-api`, then `curl -F file=@<png> localhost:8000/predict`. Record the response and the image size.
- [ ] **Step 3: Commit** `feat: Dockerfile for inference API`.

### Task 13 (optional): multi-seed + latency
Train `cnn_improved` and `resnet18_finetune` with seeds 43 and 44 and evaluate them. Extend `compare.py` to aggregate `<preset>_s*` as mean ± std. Add CPU latency (batch 1, median of 200) and throughput (batch 256) columns using `time.perf_counter` after warm-up. Commit `feat(compare): multi-seed aggregation and latency`.
