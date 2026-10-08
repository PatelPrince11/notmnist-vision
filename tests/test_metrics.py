import numpy as np
import pytest

from notmnist.metrics import compute_metrics, wilson_ci


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


def test_output_is_json_serialisable_and_shaped():
    import json
    from notmnist import CLASS_NAMES
    y = np.array([0, 1, 2, 2]); m = compute_metrics(np.eye(10)[y], y)
    json.dumps(m)
    assert list(m["per_class"]) == list(CLASS_NAMES)
    assert len(m["confusion_matrix"]) == 10 and len(m["accuracy_ci95"]) == 2
