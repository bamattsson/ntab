"""Load benchmark predictions and compute per-similarity-bin statistics."""

import pathlib
import warnings
from collections import Counter

import pandas as pd

from nfab_evaluate.metrics import pearson_r_per_assay


def _parse_split_label(split: str) -> tuple[float, float, str]:
    """Parse a test split label into (lo, hi, display_label).

    'test_sim_0.00_0.35' → (0.0, 0.35, '[0.00, 0.35)')
    'test_sim_1.00'      → (1.0, 1.0,  '=1.00')
    """
    suffix = split.removeprefix("test_sim_")
    parts = suffix.split("_")
    if len(parts) == 1:
        v = float(parts[0])
        return v, v, f"={v:.2f}"
    lo, hi = float(parts[0]), float(parts[1])
    return lo, hi, f"[{lo:.2f}, {hi:.2f})"


def load_predictions(
    pred_path: str | pathlib.Path,
    test_activities: pd.DataFrame,
) -> pd.DataFrame:
    """Load a predictions CSV and join against test_activities.

    Returns a DataFrame with all rows from test_activities, adding pred_pchembl.
    Rows not present in the CSV are filled with 6.0 and a warning is printed.

    Args:
        pred_path: Path to predictions CSV with columns:
            assay_id, ligand_name, standard_type, pred_pchembl.
        test_activities: DataFrame with columns:
            assay_chembl_id, ligand_chembl_id, standard_type,
            split, pchembl_value_filled.

    Returns:
        Merged DataFrame with pred_pchembl column added.
    """
    preds = pd.read_csv(
        pred_path,
        usecols=lambda c: (
            c in {"assay_id", "ligand_name", "standard_type", "pred_pchembl"}
        ),
    )

    merged = test_activities.merge(
        preds[["assay_id", "ligand_name", "standard_type", "pred_pchembl"]],
        left_on=["assay_chembl_id", "ligand_chembl_id", "standard_type"],
        right_on=["assay_id", "ligand_name", "standard_type"],
        how="left",
    )

    n_missing = merged["pred_pchembl"].isna().sum()
    if n_missing > 0:
        warnings.warn(
            f"{n_missing:,} of {len(merged):,} test rows have no prediction in "
            f"{pred_path}; filling pred_pchembl = 6.0"
        )
        merged["pred_pchembl"] = merged["pred_pchembl"].fillna(6.0)

    return merged


def compute_model_stats(
    merged: pd.DataFrame,
    bins: list[tuple[float, float, str]],
    min_assay_size: int = 10,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute mean Pearson r per similarity bin for a single model.

    Args:
        merged: Output of load_predictions — test activities with pred_pchembl.
        bins: List of (lo, hi, display_label) tuples as returned by
            _parse_split_label, one per similarity bin.
        min_assay_size: Minimum compounds per assay to include in Pearson r.
        n_bootstrap: Number of bootstrap resamples for the 95% CI.

    Returns:
        DataFrame with columns:
            display_label, n_m (molecules), n_a (qualifying assays),
            pearson_r, ci_low, ci_high.
    """
    records = []
    for lo, hi, display_label in bins:
        split_label = (
            f"test_sim_{lo:.2f}" if lo == hi else f"test_sim_{lo:.2f}_{hi:.2f}"
        )
        subset = merged[merged["split"] == split_label].dropna(
            subset=["pred_pchembl", "pchembl_value_filled"]
        )
        assay_keys = (
            subset["assay_chembl_id"] + "__" + subset["standard_type"]
        ).tolist()

        r, ci_low, ci_high = pearson_r_per_assay(
            preds=subset["pred_pchembl"].values,
            labels=subset["pchembl_value_filled"].values,
            assay_ids=assay_keys,
            min_assay_size=min_assay_size,
            n_bootstrap=n_bootstrap,
        )

        counts = Counter(assay_keys)
        n_a = sum(1 for c in counts.values() if c >= min_assay_size)
        records.append(
            dict(
                display_label=display_label,
                n_m=len(subset),
                n_a=n_a,
                pearson_r=r,
                ci_low=ci_low if ci_low is not None else float("nan"),
                ci_high=ci_high if ci_high is not None else float("nan"),
            )
        )
    return pd.DataFrame(records)
