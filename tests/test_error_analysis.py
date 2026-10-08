import numpy as np

from notmnist.error_analysis import top_confusions


def test_top_confusions_hand_case():
    y = np.array([0, 1, 2])
    pred = np.array([0, 2, 1])
    # diagonal excluded; tie on count broken by (true, pred) ascending
    assert top_confusions(y, pred) == [(1, 2, 1), (2, 1, 1)]


def test_top_confusions_orders_by_count_and_limits_k():
    y = np.array([3, 3, 3, 1, 1, 0])
    pred = np.array([4, 4, 4, 2, 2, 5])
    assert top_confusions(y, pred, k=2) == [(3, 4, 3), (1, 2, 2)]
