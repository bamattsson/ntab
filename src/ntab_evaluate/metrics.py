"""Evaluation metrics for the NFAB benchmark."""

import math

import numpy as np
from scipy.stats import pearsonr

MIN_ASSAY_SIZE: int = 10


def mae_per_assay(
    preds: np.ndarray,
    labels: np.ndarray,
    assay_ids: list[str],
    min_assay_size: int = MIN_ASSAY_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MAE for each qualifying assay.

    Args:
        preds: Predicted values, shape (N,).
        labels: True values, shape (N,).
        assay_ids: Assay identifier per sample, length N.
        min_assay_size: Assays with fewer than this many samples are excluded.

    Returns:
        Tuple of (assay_ids, mae, assay_sizes) as numpy arrays, one entry per
        qualifying assay. assay_ids has dtype object, mae float64, assay_sizes int64.
    """
    assay_to_indices: dict[str, list[int]] = {}
    for i, assay in enumerate(assay_ids):
        assay_to_indices.setdefault(assay, []).append(i)

    out_ids: list[str] = []
    out_mae: list[float] = []
    out_sizes: list[int] = []
    for assay, indices in sorted(assay_to_indices.items()):
        if len(indices) < min_assay_size:
            continue
        idx = np.array(indices)
        assay_mae = float(np.mean(np.abs(preds[idx] - labels[idx])))
        if not math.isfinite(assay_mae):
            continue
        out_ids.append(assay)
        out_mae.append(assay_mae)
        out_sizes.append(len(indices))

    return (
        np.array(out_ids, dtype=object),
        np.array(out_mae, dtype=np.float64),
        np.array(out_sizes, dtype=np.int64),
    )


def pearson_r_per_assay(
    preds: np.ndarray,
    labels: np.ndarray,
    assay_ids: list[str],
    min_assay_size: int = MIN_ASSAY_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Pearson r for each qualifying assay.

    Args:
        preds: Predicted values, shape (N,).
        labels: True values, shape (N,).
        assay_ids: Assay identifier per sample, length N.
        min_assay_size: Assays with fewer than this many samples are excluded.

    Returns:
        Tuple of (assay_ids, pearson_r, assay_sizes) as numpy arrays, one entry
        per qualifying assay. assay_ids has dtype object, pearson_r float64,
        assay_sizes int64. Non-finite r values are excluded.
    """
    assay_to_indices: dict[str, list[int]] = {}
    for i, assay in enumerate(assay_ids):
        assay_to_indices.setdefault(assay, []).append(i)

    out_ids: list[str] = []
    out_r: list[float] = []
    out_sizes: list[int] = []
    for assay, indices in sorted(assay_to_indices.items()):
        if len(indices) < min_assay_size:
            continue
        idx = np.array(indices)
        r, _ = pearsonr(preds[idx], labels[idx])
        # Non-finite r (e.g. constant predictions) → 0.0 to penalise the assay.
        if not math.isfinite(r):
            r = 0.0
        out_ids.append(assay)
        out_r.append(float(r))
        out_sizes.append(len(indices))

    return (
        np.array(out_ids, dtype=object),
        np.array(out_r, dtype=np.float64),
        np.array(out_sizes, dtype=np.int64),
    )


def aggregate_per_assay(
    metric: np.ndarray,
    assay_size: np.ndarray | None = None,
    n_bootstrap: int | None = None,
    seed_bootstrap: int | None = None,
    weighted: bool = False,
) -> tuple[float, float | None, float | None, float | None]:
    """Aggregate per-assay metric values to a mean with optional bootstrap CI and SE.

    Args:
        metric: Per-assay metric values (already filtered by min_assay_size),
            shape (n_assays,).
        assay_size: Sample count per assay, shape (n_assays,). Required when
            weighted=True.
        n_bootstrap: If set, compute a 95% CI and SE using this many bootstrap
            resamples (resampling at the assay level).
        seed_bootstrap: Random seed for reproducible bootstrap.
        weighted: If True, weight each assay by its sample count. Requires
            assay_size to be provided.

    Returns:
        Tuple of (mean, ci_low, ci_high, se). mean is NaN if metric is empty.
        ci_low, ci_high, and se are None when n_bootstrap is not provided.
    """
    if weighted and assay_size is None:
        raise ValueError("assay_size is required when weighted=True")

    if len(metric) == 0:
        return float("nan"), None, None, None

    if weighted:
        w = assay_size.astype(np.float64)
        mean = float((metric * w).sum() / w.sum())
    else:
        mean = float(metric.mean())

    if n_bootstrap is None:
        return mean, None, None, None

    rng = np.random.default_rng(seed=seed_bootstrap)
    n = len(metric)
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_vals = metric[boot_idx]
    if weighted:
        w = assay_size.astype(np.float64)
        boot_ws = w[boot_idx]
        boot_means = (boot_vals * boot_ws).sum(axis=1) / boot_ws.sum(axis=1)
    else:
        boot_means = boot_vals.mean(axis=1)

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    se = float(boot_means.std(ddof=1))
    return mean, float(ci_low), float(ci_high), se
