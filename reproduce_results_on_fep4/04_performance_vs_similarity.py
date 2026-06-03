"""Plot per-bin average Pearson r with bootstrapped 95% CI as a function of
mean max Tanimoto similarity to the training set.

Similarities are computed on-the-fly from ECFP4 fingerprints rather than
loaded from a pre-computed CSV.

Usage
-----
    python reproduce_results_on_fep4/04_performance_vs_similarity.py \
        --predictions old_FEP_models/pred_mol_prop_only_OpenFE.csv \
        --benchmark ../paper-identifying_and_addressing_data_leakage/data/out/OpenFE_benchmark.csv \
        --training-data out_FEP4_old_baseline/data_preprocessing/train.npz \
        --output-plot openfe_performance_vs_similarity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ntab_evaluate.metrics import aggregate_per_assay, pearson_r_per_assay
from ntab_preprocess.similarity import compute_ecfp4_fingerprints, filter_by_tanimoto


BIN_EDGES = [0.3, 0.5, 0.65, 0.8, 1.0]
N_BOOTSTRAP = 1000
SHOW_POINTS = True


def compute_max_tanimoto(
    test_names: list[str],
    test_smiles: list[str],
    train_smiles: np.ndarray,
    train_ids: np.ndarray,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute max Tanimoto similarity of each test compound vs training set.

    Args:
        test_names: Identifier per test compound (ligand_name).
        test_smiles: SMILES per test compound.
        train_smiles: SMILES array from training npz.
        train_ids: Ligand ID array from training npz.
        n_jobs: Worker processes for fingerprint computation.

    Returns:
        DataFrame with columns ligand_name, max_tanimoto_sim.
    """
    train_unique = pd.DataFrame({"ligand_id": train_ids, "smiles": train_smiles}).drop_duplicates(
        subset="ligand_id"
    )
    train_fp_names, train_fps = compute_ecfp4_fingerprints(
        train_unique["ligand_id"].tolist(),
        train_unique["smiles"].tolist(),
        n_jobs=n_jobs,
    )

    test_fp_names, test_fps = compute_ecfp4_fingerprints(
        test_names,
        test_smiles,
        n_jobs=n_jobs,
    )

    _, max_sims, _ = filter_by_tanimoto(
        candidate_fps=test_fps,
        reference_fps=train_fps,
        reference_ids=train_fp_names,
        threshold=0.0,
        n_jobs=n_jobs,
    )

    return pd.DataFrame({
        "ligand_name": test_fp_names,
        "max_tanimoto_sim": max_sims,
    })


def compute_per_assay_stats(
    pred_df: pd.DataFrame,
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-assay Pearson r and mean max Tanimoto similarity.

    Args:
        pred_df: Predictions with assay_id, ligand_name, standard_type,
            pchembl_value, pred_pchembl.
        sim_df: Per-ligand similarities with ligand_name, max_tanimoto_sim.

    Returns:
        DataFrame with one row per assay: assay_group, pearson_r, mean_max_sim,
        n_compounds.
    """
    merged = pred_df.merge(sim_df[["ligand_name", "max_tanimoto_sim"]], on="ligand_name")
    merged["assay_group"] = merged["assay_id"].astype(str) + "_" + merged["standard_type"]

    assay_keys = merged["assay_group"].tolist()
    preds = merged["pred_pchembl"].values
    labels = merged["pchembl_value"].values

    assay_ids, r_vals, sizes = pearson_r_per_assay(
        preds, labels, assay_keys, min_assay_size=2,
    )

    sim_by_assay = merged.groupby("assay_group")["max_tanimoto_sim"].mean()

    rows = []
    for aid, r, n in zip(assay_ids, r_vals, sizes):
        rows.append(
            {
                "assay_group": aid,
                "pearson_r": r,
                "mean_max_sim": sim_by_assay[aid],
                "n_compounds": int(n),
            }
        )

    return pd.DataFrame(rows)


def assign_bins(assay_df: pd.DataFrame) -> list[tuple[float, float, pd.DataFrame]]:
    """Assign assays to similarity bins.

    Returns:
        List of (lo, hi, subset_df) tuples.
    """
    bins = list(zip(BIN_EDGES[:-1], BIN_EDGES[1:]))
    result = []
    for lo, hi in bins:
        if hi == BIN_EDGES[-1]:
            mask = (assay_df["mean_max_sim"] >= lo) & (assay_df["mean_max_sim"] <= hi)
        else:
            mask = (assay_df["mean_max_sim"] >= lo) & (assay_df["mean_max_sim"] < hi)
        result.append((lo, hi, assay_df.loc[mask]))
    return result


def make_plot(
    assay_df: pd.DataFrame,
    output_path: Path,
    show_points: bool = SHOW_POINTS,
) -> None:
    """Create line plot with bootstrapped 95% CI bands.

    Args:
        assay_df: DataFrame with pearson_r, mean_max_sim, n_compounds columns.
        output_path: Where to save the plot.
        show_points: Whether to overlay individual assay data points.
    """
    binned = assign_bins(assay_df)

    bin_labels = []
    means = []
    ci_lows = []
    ci_highs = []
    n_rows_per_bin = []
    n_assays_per_bin = []
    points_per_bin: list[np.ndarray] = []

    for lo, hi, subset in binned:
        r_vals = subset["pearson_r"].to_numpy()
        sizes = subset["n_compounds"].to_numpy()
        n_rows = int(sizes.sum()) if len(sizes) > 0 else 0

        mean, ci_low, ci_high, _ = aggregate_per_assay(
            r_vals,
            assay_size=sizes,
            n_bootstrap=N_BOOTSTRAP,
            seed_bootstrap=42,
            weighted=True,
        )

        if hi == BIN_EDGES[-1]:
            bin_labels.append(f"[{lo}, {hi}]")
        else:
            bin_labels.append(f"[{lo}, {hi})")
        means.append(mean)
        ci_lows.append(ci_low if ci_low is not None else mean)
        ci_highs.append(ci_high if ci_high is not None else mean)
        n_rows_per_bin.append(n_rows)
        n_assays_per_bin.append(len(subset))
        points_per_bin.append(r_vals)

    x = np.arange(len(bin_labels))
    means = np.array(means)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    fig, ax = plt.subplots(figsize=(8, 6))

    yerr_low = means - ci_lows
    yerr_high = ci_highs - means
    ax.errorbar(x, means, yerr=[yerr_low, yerr_high], fmt="o", markersize=8,
                color="#0077BB", ecolor="#0077BB", elinewidth=2, capsize=5, capthick=2, zorder=4)

    if show_points:
        rng = np.random.default_rng(42)
        for i, pts in enumerate(points_per_bin):
            if len(pts) > 0:
                jitter = rng.uniform(-0.15, 0.15, size=len(pts))
                ax.scatter(i + jitter, pts, color="black", s=12, alpha=0.4, zorder=3)

    xticklabels = [
        f"{label}\n$n_a$={n_a}\n$n_m$={n_r}"
        for label, n_a, n_r in zip(bin_labels, n_assays_per_bin, n_rows_per_bin)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=11)
    ax.set_xlabel("Mean maximum Tanimoto similarity to the training set",
                  fontsize=11)
    ax.set_ylabel("Mean Pearson Correlation", fontsize=11)

    ax.set_title("FEP+ OpenFE: performance vs. similarity", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def run_tukey_hsd(assay_df: pd.DataFrame) -> None:
    """Run pairwise Tukey HSD test on per-assay Pearson r across similarity bins."""
    binned = assign_bins(assay_df)

    labels = []
    values = []
    for lo, hi, subset in binned:
        bin_label = f"{lo}-{hi}"
        for r in subset["pearson_r"]:
            labels.append(bin_label)
            values.append(r)

    if len(set(labels)) < 2:
        print("\nTukey HSD: fewer than 2 non-empty bins, skipping.")
        return

    result = pairwise_tukeyhsd(endog=values, groups=labels)
    print("\n=== Tukey HSD (per-assay Pearson r across similarity bins) ===")
    print(result.summary())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-bin Pearson r with bootstrap CI vs. Tanimoto similarity."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Predictions CSV (assay_id, ligand_name, standard_type, pchembl_value, pred_pchembl)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark CSV with ligand_name and smiles columns (for SMILES lookup)",
    )
    parser.add_argument(
        "--training-data",
        required=True,
        help="Path to train.npz with smiles and ligand_ids arrays",
    )
    parser.add_argument(
        "--output-plot",
        default="openfe_performance_vs_similarity.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--no-points",
        action="store_true",
        default=False,
        help="Hide individual assay data points",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of worker processes for fingerprint computation (default: 1, -1 for all CPUs)",
    )
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions)
    benchmark_df = pd.read_csv(args.benchmark, usecols=["ligand_name", "smiles"])
    train_data = np.load(args.training_data, allow_pickle=True)

    test_ligands = pred_df[["ligand_name"]].drop_duplicates().merge(
        benchmark_df, on="ligand_name", how="left",
    )
    missing = test_ligands["smiles"].isna().sum()
    if missing > 0:
        raise ValueError(f"{missing} test ligands have no SMILES in benchmark CSV")

    print(f"Computing ECFP4 fingerprints and max Tanimoto similarity "
          f"({len(test_ligands)} test vs {len(np.unique(train_data['ligand_ids']))} train compounds)...")
    sim_df = compute_max_tanimoto(
        test_names=test_ligands["ligand_name"].tolist(),
        test_smiles=test_ligands["smiles"].tolist(),
        train_smiles=train_data["smiles"],
        train_ids=train_data["ligand_ids"],
        n_jobs=args.n_jobs,
    )

    assay_df = compute_per_assay_stats(pred_df, sim_df)

    print(f"\nPer-assay stats ({len(assay_df)} assays):")
    for _, row in assay_df.sort_values("mean_max_sim").iterrows():
        print(
            f"  {row['assay_group']:40s}  r={row['pearson_r']:+.4f}"
            f"  mean_sim={row['mean_max_sim']:.4f}  n={row['n_compounds']}"
        )

    make_plot(
        assay_df,
        Path(args.output_plot),
        show_points=not args.no_points,
    )

    run_tukey_hsd(assay_df)


if __name__ == "__main__":
    main()
