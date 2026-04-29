import math

import numpy as np
import pytest

from nfab_evaluate.metrics import aggregate_per_assay, mae_per_assay, pearson_r_per_assay


def _make_data(n: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    labels = rng.normal(size=n)
    preds = labels + rng.normal(scale=0.3, size=n)
    assay_ids = [f"assay_{i // 10}" for i in range(n)]
    return preds, labels, assay_ids


# ---------------------------------------------------------------------------
# pearson_r_per_assay
# ---------------------------------------------------------------------------


def test_pearson_r_returns_three_arrays():
    preds, labels, assay_ids = _make_data()
    result = pearson_r_per_assay(preds, labels, assay_ids)
    assert len(result) == 3
    ids, r_vals, sizes = result
    assert isinstance(ids, np.ndarray)
    assert isinstance(r_vals, np.ndarray)
    assert isinstance(sizes, np.ndarray)
    assert ids.shape == r_vals.shape == sizes.shape


def test_pearson_r_correct_value():
    # Perfect correlation → r = 1.0
    vals = np.arange(10, dtype=float)
    assay_ids = ["A"] * 10
    ids, r_vals, sizes = pearson_r_per_assay(vals, vals, assay_ids, min_assay_size=5)
    assert len(ids) == 1
    assert ids[0] == "A"
    assert pytest.approx(r_vals[0], abs=1e-5) == 1.0
    assert sizes[0] == 10


def test_pearson_r_min_assay_size_filters():
    # Assay A: 10 samples (qualifies), Assay B: 5 samples (filtered out at default=10)
    preds = np.concatenate([np.arange(10, dtype=float), np.arange(5, dtype=float)])
    labels = preds.copy()
    assay_ids = ["A"] * 10 + ["B"] * 5
    ids, r_vals, sizes = pearson_r_per_assay(preds, labels, assay_ids)
    assert list(ids) == ["A"]
    assert sizes[0] == 10


def test_pearson_r_empty_when_none_qualify():
    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 2.0, 3.0])
    assay_ids = ["A", "A", "A"]
    ids, r_vals, sizes = pearson_r_per_assay(preds, labels, assay_ids, min_assay_size=10)
    assert len(ids) == 0
    assert len(r_vals) == 0
    assert len(sizes) == 0


def test_pearson_r_multiple_assays():
    # Assay A: r=1, Assay B: r=-1, both with 10 samples
    preds_a = np.arange(10, dtype=float)
    preds_b = np.arange(10, dtype=float)[::-1].copy()
    labels_ab = np.arange(10, dtype=float)
    preds = np.concatenate([preds_a, preds_b])
    labels = np.concatenate([labels_ab, labels_ab])
    assay_ids = ["A"] * 10 + ["B"] * 10
    ids, r_vals, sizes = pearson_r_per_assay(preds, labels, assay_ids, min_assay_size=5)
    assert set(ids) == {"A", "B"}
    r_by_id = dict(zip(ids, r_vals))
    assert pytest.approx(r_by_id["A"], abs=1e-5) == 1.0
    assert pytest.approx(r_by_id["B"], abs=1e-5) == -1.0


# ---------------------------------------------------------------------------
# mae_per_assay
# ---------------------------------------------------------------------------


def test_mae_returns_three_arrays():
    preds, labels, assay_ids = _make_data()
    ids, mae_vals, sizes = mae_per_assay(preds, labels, assay_ids)
    assert ids.shape == mae_vals.shape == sizes.shape


def test_mae_perfect_predictions():
    labels = np.arange(10, dtype=float)
    assay_ids = ["A"] * 10
    ids, mae_vals, sizes = mae_per_assay(labels, labels, assay_ids, min_assay_size=5)
    assert len(ids) == 1
    assert pytest.approx(mae_vals[0]) == 0.0


def test_mae_known_value():
    # Assay A: errors all 1.0 → MAE = 1.0
    labels = np.zeros(10)
    preds = np.ones(10)
    assay_ids = ["A"] * 10
    ids, mae_vals, sizes = mae_per_assay(preds, labels, assay_ids, min_assay_size=5)
    assert pytest.approx(mae_vals[0]) == 1.0


def test_mae_min_assay_size_filters():
    preds = np.array([1.0, 2.0, 3.0])
    labels = np.zeros(3)
    assay_ids = ["A", "A", "A"]
    ids, mae_vals, sizes = mae_per_assay(preds, labels, assay_ids, min_assay_size=10)
    assert len(ids) == 0


# ---------------------------------------------------------------------------
# aggregate_per_assay
# ---------------------------------------------------------------------------


def test_aggregate_basic_mean():
    metric = np.array([0.2, 0.4, 0.6])
    mean, ci_low, ci_high = aggregate_per_assay(metric)
    assert pytest.approx(mean) == 0.4
    assert ci_low is None
    assert ci_high is None


def test_aggregate_empty_returns_nan():
    mean, ci_low, ci_high = aggregate_per_assay(np.array([]))
    assert math.isnan(mean)
    assert ci_low is None
    assert ci_high is None


def test_aggregate_weighted_requires_sizes():
    metric = np.array([0.2, 0.8])
    with pytest.raises(ValueError, match="assay_size"):
        aggregate_per_assay(metric, weighted=True)


def test_aggregate_weighted_mean():
    # Assay A: r=0.0, n=10; Assay B: r=1.0, n=30 → weighted mean = 0.75
    metric = np.array([0.0, 1.0])
    sizes = np.array([10, 30])
    mean, _, _ = aggregate_per_assay(metric, assay_size=sizes, weighted=True)
    assert pytest.approx(mean) == 0.75


def test_aggregate_bootstrap_ci_ordered():
    rng = np.random.default_rng(0)
    metric = rng.uniform(0, 1, size=20)
    mean, ci_low, ci_high = aggregate_per_assay(metric, n_bootstrap=500, seed_bootstrap=42)
    assert ci_low is not None and ci_high is not None
    assert ci_low <= mean <= ci_high
