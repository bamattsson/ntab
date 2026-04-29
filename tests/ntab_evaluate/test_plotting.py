import io
import pathlib

import numpy as np
import pandas as pd
import pytest

from ntab_evaluate.plotting import (
    _parse_split_label,
    compute_model_stats,
    load_predictions,
)


# ---------------------------------------------------------------------------
# _parse_split_label
# ---------------------------------------------------------------------------


def test_parse_range_bin():
    lo, hi, label = _parse_split_label("test_sim_0.00_0.35")
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.35)
    assert label == "[0.00, 0.35)"


def test_parse_exact_bin():
    lo, hi, label = _parse_split_label("test_sim_1.00")
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)
    assert label == "=1.00"


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------


def _make_activities(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "assay_chembl_id": f"CHEMBL_A{i // 10}",
                "ligand_chembl_id": f"CHEMBL_L{i}",
                "standard_type": "IC50",
                "split": "test_sim_0.00_0.35",
                "pchembl_value_filled": float(rng.uniform(5, 10)),
            }
        )
    return pd.DataFrame(rows)


def _make_pred_csv(activities: pd.DataFrame, missing: int = 0) -> pathlib.Path:
    rows = []
    for _, r in activities.iloc[: len(activities) - missing].iterrows():
        rows.append(
            {
                "assay_id": r["assay_chembl_id"],
                "ligand_name": r["ligand_chembl_id"],
                "standard_type": r["standard_type"],
                "pred_pchembl": r["pchembl_value_filled"] + 0.1,
            }
        )
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf  # type: ignore[return-value]


def test_load_predictions_full_match():
    activities = _make_activities()
    csv_buf = _make_pred_csv(activities)
    merged = load_predictions(csv_buf, activities)
    assert len(merged) == len(activities)
    assert merged["pred_pchembl"].isna().sum() == 0


def test_load_predictions_missing_filled_with_6():
    activities = _make_activities(30)
    csv_buf = _make_pred_csv(activities, missing=5)
    with pytest.warns(UserWarning, match="5"):
        merged = load_predictions(csv_buf, activities)
    assert (merged["pred_pchembl"] == 6.0).sum() == 5


# ---------------------------------------------------------------------------
# compute_model_stats
# ---------------------------------------------------------------------------


def _make_merged(
    n_per_bin: int = 20,
) -> tuple[pd.DataFrame, list[tuple[float, float, str]]]:
    rng = np.random.default_rng(1)
    bins = [(0.0, 0.35, "[0.00, 0.35)"), (0.35, 0.50, "[0.35, 0.50)")]
    rows = []
    for lo, hi, _ in bins:
        split_label = f"test_sim_{lo:.2f}_{hi:.2f}"
        for i in range(n_per_bin):
            rows.append(
                {
                    "assay_chembl_id": f"A{i // 10}",
                    "standard_type": "IC50",
                    "split": split_label,
                    "pchembl_value_filled": float(rng.uniform(5, 10)),
                    "pred_pchembl": float(rng.uniform(5, 10)),
                }
            )
    return pd.DataFrame(rows), bins


def test_compute_model_stats_returns_two_dataframes():
    merged, bins = _make_merged()
    result = compute_model_stats(merged, bins, min_assay_size=5, n_bootstrap=50)
    assert isinstance(result, tuple) and len(result) == 2
    assert all(isinstance(df, pd.DataFrame) for df in result)


def test_compute_model_stats_aggregated_shape():
    merged, bins = _make_merged()
    _, aggregated = compute_model_stats(merged, bins, min_assay_size=5, n_bootstrap=50)
    assert len(aggregated) == len(bins)
    assert set(aggregated.columns) == {
        "display_label",
        "n_m",
        "n_a",
        "pearson_r",
        "pearson_r_ci_low",
        "pearson_r_ci_high",
        "mae",
        "mae_ci_low",
        "mae_ci_high",
    }


def test_compute_model_stats_aggregated_n_m():
    merged, bins = _make_merged(n_per_bin=20)
    _, aggregated = compute_model_stats(merged, bins, min_assay_size=5, n_bootstrap=50)
    assert (aggregated["n_m"] == 20).all()


def test_compute_model_stats_per_assay_columns():
    merged, bins = _make_merged()
    per_assay, _ = compute_model_stats(merged, bins, min_assay_size=5, n_bootstrap=50)
    assert {"assay_id", "bin_label", "display_label", "pearson_r", "mae", "n"}.issubset(
        per_assay.columns
    )


def test_compute_model_stats_per_assay_bin_labels():
    merged, bins = _make_merged()
    per_assay, _ = compute_model_stats(merged, bins, min_assay_size=5, n_bootstrap=50)
    assert set(per_assay["bin_label"].unique()) == {
        f"test_sim_{lo:.2f}_{hi:.2f}" for lo, hi, _ in bins
    }
