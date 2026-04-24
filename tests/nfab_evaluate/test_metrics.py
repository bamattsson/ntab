import math

import numpy as np

from nfab_evaluate.metrics import mae_per_assay, pearson_r_per_assay


def _make_data(n: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    labels = rng.normal(size=n)
    preds = labels + rng.normal(scale=0.3, size=n)
    assay_ids = [f"assay_{i // 10}" for i in range(n)]
    return preds, labels, assay_ids


def test_basic_returns_finite():
    preds, labels, assay_ids = _make_data()
    r, ci_low, ci_high = pearson_r_per_assay(preds, labels, assay_ids)
    assert math.isfinite(r)
    assert 0.0 < r <= 1.0
    assert ci_low is None
    assert ci_high is None


def test_below_min_assay_size_returns_nan():
    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 2.0, 3.0])
    assay_ids = ["a", "a", "a"]
    r, ci_low, ci_high = pearson_r_per_assay(
        preds, labels, assay_ids, min_assay_size=10
    )
    assert math.isnan(r)
    assert ci_low is None
    assert ci_high is None


def test_bootstrap_ci_bounds():
    preds, labels, assay_ids = _make_data(n=40)
    r, ci_low, ci_high = pearson_r_per_assay(preds, labels, assay_ids, n_bootstrap=200)
    assert ci_low is not None and ci_high is not None
    assert ci_low <= r <= ci_high


def test_weighted_differs_from_macro():
    rng = np.random.default_rng(42)
    # Two assays with very different sizes and correlations
    n_small, n_large = 10, 50
    labels_small = rng.normal(size=n_small)
    preds_small = -labels_small  # r ≈ -1
    labels_large = rng.normal(size=n_large)
    preds_large = labels_large  # r ≈ +1
    preds = np.concatenate([preds_small, preds_large])
    labels = np.concatenate([labels_small, labels_large])
    assay_ids = ["small"] * n_small + ["large"] * n_large

    r_macro, _, _ = pearson_r_per_assay(preds, labels, assay_ids, weighted=False)
    r_weighted, _, _ = pearson_r_per_assay(preds, labels, assay_ids, weighted=True)

    # macro average is closer to 0 (equal weight); weighted skews toward large assay (r≈+1)
    assert r_weighted > r_macro


# ---------------------------------------------------------------------------
# mae_per_assay
# ---------------------------------------------------------------------------


def test_mae_perfect_predictions():
    labels = np.array([1.0] * 10 + [2.0] * 10, dtype=float)
    preds = labels.copy()
    assay_ids = ["a"] * 10 + ["b"] * 10
    mae, ci_low, ci_high = mae_per_assay(preds, labels, assay_ids)
    assert mae == 0.0
    assert ci_low is None


def test_mae_known_value():
    # assay_a: errors all 1.0 → MAE = 1.0
    # assay_b: errors all 2.0 → MAE = 2.0
    # macro MAE = 1.5
    labels = np.zeros(20)
    preds = np.array([1.0] * 10 + [2.0] * 10)
    assay_ids = ["a"] * 10 + ["b"] * 10
    mae, _, _ = mae_per_assay(preds, labels, assay_ids)
    assert math.isclose(mae, 1.5)


def test_mae_below_min_assay_size_returns_nan():
    preds = np.array([1.0, 2.0, 3.0])
    labels = np.zeros(3)
    assay_ids = ["a", "a", "a"]
    mae, ci_low, ci_high = mae_per_assay(preds, labels, assay_ids, min_assay_size=10)
    assert math.isnan(mae)
    assert ci_low is None


def test_mae_bootstrap_ci_ordered():
    preds, labels, assay_ids = _make_data(n=40)
    mae, ci_low, ci_high = mae_per_assay(preds, labels, assay_ids, n_bootstrap=200)
    assert ci_low is not None and ci_high is not None
    assert ci_low <= mae <= ci_high
