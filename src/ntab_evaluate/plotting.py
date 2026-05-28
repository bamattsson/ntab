"""Load benchmark predictions and compute per-similarity-bin statistics."""

import pathlib
import warnings

import pandas as pd

from ntab_evaluate.metrics import (
    aggregate_per_assay,
    mae_per_assay,
    pearson_r_per_assay,
)


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-bin Pearson r and MAE for a single model.

    Args:
        merged: Output of load_predictions — test activities with pred_pchembl.
        bins: List of (lo, hi, display_label) tuples as returned by
            _parse_split_label, one per similarity bin.
        min_assay_size: Minimum compounds per assay to include in metrics.
        n_bootstrap: Number of bootstrap resamples for the 95% CI.

    Returns:
        Tuple of (per_assay, aggregated).

        per_assay: One row per qualifying (assay, bin) pair. Columns:
            assay_id, bin_label, display_label, pearson_r, mae, n.
        aggregated: One row per similarity bin. Columns:
            display_label, n_m, n_a, pearson_r, pearson_r_ci_low,
            pearson_r_ci_high, mae, mae_ci_low, mae_ci_high.
    """
    per_assay_records = []
    agg_records = []

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
        preds = subset["pred_pchembl"].values
        labels = subset["pchembl_value_filled"].values

        ids_r, r_vals, sizes_r = pearson_r_per_assay(
            preds, labels, assay_keys, min_assay_size
        )
        ids_mae, mae_vals, _ = mae_per_assay(preds, labels, assay_keys, min_assay_size)

        r_by_id = dict(zip(ids_r, r_vals))
        mae_by_id = dict(zip(ids_mae, mae_vals))
        size_by_id = dict(zip(ids_r, sizes_r))

        all_ids = sorted(set(ids_r) | set(ids_mae))
        for aid in all_ids:
            per_assay_records.append(
                {
                    "assay_id": aid,
                    "bin_label": split_label,
                    "display_label": display_label,
                    "pearson_r": r_by_id.get(aid, float("nan")),
                    "mae": mae_by_id.get(aid, float("nan")),
                    "n": size_by_id.get(aid, 0),
                }
            )

        r, r_ci_low, r_ci_high, _ = aggregate_per_assay(
            r_vals, n_bootstrap=n_bootstrap, seed_bootstrap=42
        )
        mae, mae_ci_low, mae_ci_high, _ = aggregate_per_assay(
            mae_vals, n_bootstrap=n_bootstrap, seed_bootstrap=42
        )

        n_a = len(ids_r)  # pearson_r_per_assay already applies min_assay_size
        agg_records.append(
            {
                "display_label": display_label,
                "n_m": len(subset),
                "n_a": n_a,
                "pearson_r": r,
                "pearson_r_ci_low": r_ci_low if r_ci_low is not None else float("nan"),
                "pearson_r_ci_high": r_ci_high
                if r_ci_high is not None
                else float("nan"),
                "mae": mae,
                "mae_ci_low": mae_ci_low if mae_ci_low is not None else float("nan"),
                "mae_ci_high": mae_ci_high if mae_ci_high is not None else float("nan"),
            }
        )

    return pd.DataFrame(per_assay_records), pd.DataFrame(agg_records)
