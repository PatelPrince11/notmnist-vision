# Legacy coursework (CPSC 433, Fall 2025)

Original Keras / TensorFlow assignment code, preserved as submitted. It is read-only history:
the maintained PyTorch work lives in `src/notmnist/`.

## Running

The scripts expect to be run from their own directory (they load files by relative path), inside
the legacy TensorFlow environment (`env/`). `notMNIST.npz` now lives in `data/` at the repo root,
so copy or symlink it next to the Part 1 scripts before running them.

- `keras_part1/`: notMNIST letter classification (Part 1)
- `nfl_draft_part2/`: NFL draft tabular model (Part 2, unrelated to computer vision)

## Part 1 models (audited 2026-10-07)

| | MLP ("Partial") | CNN ("Complete") |
|---|---|---|
| Architecture | Flatten, Dense 256 ReLU, Dense 10 softmax | Conv 32 3x3, MaxPool 2, Conv 64 3x3, MaxPool 2, Flatten 1600, Dense 256, Dropout 0.4, Dense 128, Dropout 0.2, Dense 10 softmax |
| Trainable params | 203,530 | 462,858 |
| Loss / optimizer | sparse CE / Adam (lr 1e-3) | sparse CE / Adam (lr 1e-3) |
| Batch / epochs | 64 / 5 | 32 / 50 (no early stopping) |
| Preprocessing | `x / 255.0`, add channel dim | same |
| Seed | `tf.random.set_seed(1234)` (not fully deterministic) | same |
| Train acc | 90.31 % | 99.27 % |
| Test acc | 93.10 % (690 errors) | 95.12 % (488 errors) |
| Test acc, de-duplicated 9,487 | 92.82 % | 94.91 % |
| Macro-F1 (test) | 0.931 | 0.951 |
| Weakest class | I (F1 0.897; 53 I to J errors) | I (F1 0.926) |

## Part 2 (tabular)

`draft.csv`: 671 players, one-hot `position` plus 11 combine stats, label `drafted` (413 yes / 258 no).
85/15 split (`random_state=42`), StandardScaler fit on train only. MLP: Dense 256 (L2 0.01), BN,
Dropout 0.3, Dense 128 (L2 0.01), BN, Dropout 0.2, Dense 1 sigmoid; Adam lr 5e-4, batch 16,
up to 100 epochs, balanced class weights, EarlyStopping on val accuracy.

## Known issues (kept as-is)

- CNN overfits: 99.27 % train vs 95.12 % test. The submitted report claims the opposite.
- Part 2 leakage: `validation_data=(X_test, y_test)`, so early stopping uses the test set.
- Part 2 encodes missing values as 0 (e.g. 355/671 `threecone`), and has no seed:
  reruns gave 74-77 % test accuracy; always predicting "drafted" scores 65.3 %.
- The report describes notMNIST as handwritten digits; it is font-rendered letters A-J.
- 513 test images duplicate training images; train has 1,761 duplicates.

## Attribution

- `predict.py`, `predict_test.py`, `interactive.py` and `grabimage.py` are the CPSC 433 course
  template by Jonathan Hudson and keep his header (`Original Author: Jonathan Hudson`, `CPSC 433 F24`).
- `notMNIST-Partial.py` and `notMNIST-Complete.py` are student-modified versions of his starter
  training script. Their header names Prince Patel and, directly below, credits the origin:
  "Based on the CPSC 433 F24 starter script by Jonathan Hudson; architecture and training
  configuration by Prince Patel." (This line, added 2026-10-07, is the only edit to the legacy code.)
- `find_misclassified.py`, `compare_partial_complete.py`, all of Part 2, and the written analysis
  are the student's own work.
