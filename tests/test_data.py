import numpy as np
import pytest
import torch

from notmnist.data import image_hashes, make_splits, to_tensor


@pytest.fixture(scope="module")
def splits():
    return make_splits(42)


def test_counts(splits):
    i = splits.info
    assert i["n_train_raw"] == 60000 and i["n_test"] == 10000
    assert i["n_train_dedup"] == 58239  # audited
    assert i["n_train"] + i["n_val"] == 58239
    assert i["n_test_clean"] == 10000 - 513  # audited


def test_no_hash_overlap(splits):
    tr, va = set(image_hashes(splits.x_train)), set(image_hashes(splits.x_val))
    te_clean = set(image_hashes(splits.x_test[splits.test_clean_mask]))
    assert not tr & va
    assert not (tr | va) & te_clean


def test_val_is_stratified(splits):
    # Dedup leaves classes unequal (class I is smaller), so stratified means
    # proportional: each class's val share matches its share of the deduped data.
    val = np.bincount(splits.y_val, minlength=10)
    total = val + np.bincount(splits.y_train, minlength=10)
    expected = total * len(splits.y_val) / total.sum()
    assert np.abs(val - expected).max() <= 1


def test_deterministic():
    a, b = make_splits(42), make_splits(42)
    assert np.array_equal(a.y_val, b.y_val) and np.array_equal(a.x_train[:50], b.x_train[:50])


def test_to_tensor_range_and_shape(splits):
    t = to_tensor(splits.x_val[:4])
    assert t.shape == (4, 1, 28, 28) and t.dtype == torch.float32 and 0 <= t.min() and t.max() <= 1
