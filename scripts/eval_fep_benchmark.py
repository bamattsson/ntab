"""Evaluate predictions against the FEP+ benchmark.

Computes per-assay Pearson r and size-weighted mean across assays.

Usage:
    uv run python eval_fep_benchmark.py \
        --predictions predictions.csv \
        --benchmark ../paper_data_leakage_FEPp_benchmark/data/out/FEPp_benchmark.csv
"""

import argparse

import pandas as pd
from scipy.stats import pearsonr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="CSV with ligand_name and pred_pchembl columns")
    parser.add_argument("--benchmark", required=True, help="FEP+ benchmark CSV")
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions)[["ligand_name", "pred_pchembl"]]
    bench = pd.read_csv(args.benchmark)[["ligand_name", "folder_name", "pchembl_value"]]

    # Note: predictions are all computed in IC50 mode even when the benchmark
    # measurement_type is Ki or Kd. In pchembl space (-log10 M), the difference
    # between IC50 and Ki/Kd is a fixed additive constant per compound type, so
    # relative rankings within an assay are preserved and Pearson r is unaffected.

    df = bench.merge(preds, on="ligand_name", how="inner")
    n_matched = len(df)
    n_bench = len(bench)
    if n_matched < n_bench:
        print(f"Warning: {n_bench - n_matched} benchmark compounds not found in predictions, skipped.\n")

    results = []
    for assay, group in df.groupby("folder_name"):
        r, _ = pearsonr(group["pchembl_value"], group["pred_pchembl"])
        results.append({"assay": assay, "n": len(group), "pearson_r": r})
        print(f"  {assay}: n={len(group)}, r={r:.3f}")

    total_n = sum(x["n"] for x in results)
    weighted_mean = sum(x["n"] * x["pearson_r"] for x in results) / total_n
    print(f"\nSize-weighted mean Pearson r: {weighted_mean:.3f}  (n={total_n})")


if __name__ == "__main__":
    main()
