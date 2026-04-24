"""Evaluation metrics for the NFAB benchmark."""

import math

import numpy as np
from scipy.stats import pearsonr

MIN_ASSAY_SIZE: int = 10


def pearson_r_per_assay(
    preds: np.ndarray,
    labels: np.ndarray,
    assay_ids: list[str],
    min_assay_size: int = MIN_ASSAY_SIZE,
    n_bootstrap: None | int = None,
    weighted: bool = False,
) -> tuple[float, float | None, float | None]:
    """Compute mean Pearson r across assays, skipping assays below min_assay_size.

    By default computes the macro-average (each qualifying assay contributes
    equally regardless of size). Pass weighted=True for size-weighted averaging.

    Args:
        preds: Predicted values, shape (N,).
        labels: True values, shape (N,).
        assay_ids: Assay identifier per sample, length N.
        min_assay_size: Assays with fewer than this many samples are excluded.
        n_bootstrap: If set, compute a 95% bootstrap confidence interval using
            this many resamples (resampling at the assay level).
        weighted: If False (default), compute macro-average (equal weight per
            assay). If True, weight each assay's Pearson r by its sample count.

    Returns:
        Tuple of (pearson_r, ci_low, ci_high) where pearson_r is the mean Pearson r
        across qualifying assays (NaN if none qualify), and ci_low/ci_high are the
        2.5th and 97.5th bootstrap percentiles, or None if n_bootstrap was not provided.
    """
    assay_to_indices: dict[str, list[int]] = {}
    for i, assay in enumerate(assay_ids):
        assay_to_indices.setdefault(assay, []).append(i)

    rs: list[float] = []
    weights: list[int] = []
    for assay, indices in sorted(assay_to_indices.items()):
        if len(indices) < min_assay_size:
            continue
        idx = np.array(indices)
        r, _ = pearsonr(preds[idx], labels[idx])
        if not math.isfinite(r):
            continue
        rs.append(r)
        weights.append(len(indices))

    if not rs:
        return float("nan"), None, None

    rs_a = np.array(rs)
    w_a = np.array(weights, dtype=np.float64)
    if weighted:
        pearson_r = float((rs_a * w_a).sum() / w_a.sum())
    else:
        pearson_r = float(rs_a.mean())

    if n_bootstrap is None:
        return pearson_r, None, None

    # Bootstrap a 95% CI by resampling assays
    rng = np.random.default_rng()
    num_assays = len(rs_a)
    boot_idx = rng.integers(0, num_assays, size=(n_bootstrap, num_assays))
    boot_rs = rs_a[boot_idx]  # (n_bootstrap, num_assays)
    if weighted:
        boot_ws = w_a[boot_idx]  # (n_bootstrap, num_assays)
        boot_means = (boot_rs * boot_ws).sum(axis=1) / boot_ws.sum(axis=1)
    else:
        boot_means = boot_rs.mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return pearson_r, float(ci_low), float(ci_high)
