import numpy as np

from notmnist import CLASS_NAMES as L
from notmnist.error_analysis import build_markdown, error_deltas, top_confusions


def test_top_confusions_hand_case():
    y = np.array([0, 1, 2])
    pred = np.array([0, 2, 1])
    # diagonal excluded; tie on count broken by (true, pred) ascending
    assert top_confusions(y, pred) == [(1, 2, 1), (2, 1, 1)]


def test_top_confusions_orders_by_count_and_limits_k():
    y = np.array([3, 3, 3, 1, 1, 0])
    pred = np.array([4, 4, 4, 2, 2, 5])
    assert top_confusions(y, pred, k=2) == [(3, 4, 3), (1, 2, 2)]


def test_error_deltas_orientation():
    y = np.array([0, 0, 0, 0])
    base = np.array([1, 1, 1, 0])  # 3 wrong
    best = np.array([0, 1, 0, 1])  # 2 wrong; fixes idx 0,2; breaks idx 3
    assert error_deltas(y, base, best) == {"fixed": 2, "introduced": 1, "n_err_base": 3, "n_err_best": 2}


def test_build_markdown_uses_best_model():
    y = np.array([0, 0, 0, 0, 1])
    base = np.array([1, 1, 1, 1, 1])   # 4 errors, all A->B
    best = np.array([0, 0, 0, 2, 1])   # 1 error, A->C
    f1 = lambda v: {c: {"f1": v} for c in L}  # noqa: E731
    md = build_markdown("BEST", "BASE", y, pred_base=base, pred_best=best, n=5,
                        f1_base=f1(0.5), f1_best=f1(0.9), tab={})
    assert "Best-model errors: 1." in md
    assert "| A | C | 1 | 100.0% |" in md and "| A | B |" not in md
    assert "| A | 0.500 | 0.900 | +0.400 |" in md
    assert "fixed 3" in md and "introduced 0" in md
