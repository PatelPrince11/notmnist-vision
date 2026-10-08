import json

import numpy as np
import pytest

from notmnist.compare import build_rows, mcnemar_exact, render_markdown


def test_mcnemar_identical():
    c = np.array([True, False, True])
    assert mcnemar_exact(c, c)["p_value"] == 1.0


def test_mcnemar_known_value():
    a = np.ones(10, dtype=bool)
    b = np.zeros(10, dtype=bool)
    r = mcnemar_exact(a, b)
    assert r["b10"] == 10 and r["b01"] == 0 and r["p_value"] == pytest.approx(2 * 0.5**10)


def _split(acc, n_params, val=True):
    d = {"n": 10, "accuracy": acc, "accuracy_ci95": [acc - 0.01, acc + 0.01],
         "macro_f1": acc, "nll": 0.5, "ece": 0.02}
    return d


def _make_run(root, name, acc, val):
    d = root / name
    d.mkdir()
    m = {"run": name, "model_name": "m", "n_params": 1234567,
         "val": _split(0.9, 1) if val else None,
         "test_clean": _split(acc, 1), "test_official": _split(acc - 0.05, 1)}
    (d / "metrics.json").write_text(json.dumps(m))
    np.save(d / "test_probs.npy", np.zeros((4, 10), dtype=np.float32))


def test_build_rows_filters_sorts_and_nulls(tmp_path):
    _make_run(tmp_path, "hi", 0.95, True)
    _make_run(tmp_path, "lo", 0.80, False)
    _make_run(tmp_path, "hi_sub512", 0.5, True)
    (tmp_path / "inprogress").mkdir()
    rows = build_rows(tmp_path)
    assert [r["run"] for r in rows] == ["lo", "hi"]
    assert rows[0]["val_acc"] is None and rows[1]["val_acc"] == 0.9
    assert rows[1]["test_clean_ci95"] == [0.94, 0.96]
    md = render_markdown(rows, [{"a": "lo", "b": "hi", "b01": 3, "b10": 1, "p_value": 0.625}], 10, 20)
    assert "—" in md and "1,234,567" in md and "95.00%" in md and "[94.00–96.00]" in md
    assert "b01" in md and "0.625" in md


def test_build_rows_missing_runs_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="runs directory not found"):
        build_rows(tmp_path / "nope")
