import numpy as np
import pytest

from churn.evaluation import best_f1_threshold, bootstrap_intervals, classification_metrics


def test_majority_class_guess_looks_fine_on_accuracy_only():
    # the reason accuracy alone is not enough for this problem
    y = np.array([1] * 27 + [0] * 73)
    always_no = np.zeros(100)
    m = classification_metrics(y, always_no, threshold=0.5)
    assert m["accuracy"] == pytest.approx(0.73)
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_classification_metrics_counts():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.2, 0.8, 0.6, 0.1])
    m = classification_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_threshold_lands_between_well_separated_classes():
    rng = np.random.default_rng(0)
    y = np.array([0] * 300 + [1] * 100)
    p = np.concatenate([rng.uniform(0.0, 0.4, 300), rng.uniform(0.6, 1.0, 100)])
    thr = best_f1_threshold(y, p)
    assert 0.35 <= thr <= 0.65
    assert classification_metrics(y, p, thr)["f1"] == pytest.approx(1.0)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 400), 0, 1)
    point = classification_metrics(y, p, 0.5)["roc_auc"]
    lo, hi = bootstrap_intervals(y, p, 0.5, n_boot=200, seed=0)["roc_auc"]
    assert lo < point < hi


def test_threshold_search_refuses_degenerate_input():
    with pytest.raises(ValueError):
        best_f1_threshold(np.array([0, 1, 0, 1]), np.full(4, 0.3))
    with pytest.raises(ValueError):
        best_f1_threshold(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))
