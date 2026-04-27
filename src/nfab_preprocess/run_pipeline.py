"""
Pipeline entry point for generating the timesplit affinity benchmark.

Usage:
    python src/nfab_preprocess/run_pipeline.py --config config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfab_preprocess.affinity_utils import add_pchembl_columns
from nfab_preprocess.assay_filter import filter_assay_types
from nfab_preprocess.chembl_requester import ChEMBLRequester
from nfab_preprocess.config import SimilarityBin, load_config
from nfab_preprocess.similarity import (
    compute_ecfp4_fingerprints,
    compute_similarity_for_cutoff_year,
)


def _bin_label(b: SimilarityBin) -> str:
    """Return the similarity bin label suffix, e.g. 'sim_0.00_0.35' or 'sim_1.00'."""
    if b.equal is not None:
        return f"sim_{b.equal:.2f}"
    return f"sim_{b.low:.2f}_{b.hi:.2f}"


def _sim_matches_bin(sim: pd.Series, b: SimilarityBin) -> pd.Series:
    """Return a boolean mask selecting rows whose similarity falls in the given bin."""
    if b.equal is not None:
        return sim == b.equal
    return (sim >= b.low) & (sim < b.hi)


def _assay_mean_sim(
    activities_subset: pd.DataFrame,
    sim_per_cpd: pd.Series,
) -> pd.Series:
    """Return per-assay mean of per-compound max similarity.

    Args:
        activities_subset: Rows to consider; must have ligand_chembl_id and assay_chembl_id.
        sim_per_cpd: Series mapping ligand_chembl_id → max similarity (from compounds_df).

    Returns:
        Series indexed by assay_chembl_id with the mean max similarity across all
        compounds in that assay.
    """
    cpd_sim = activities_subset["ligand_chembl_id"].map(sim_per_cpd)
    return cpd_sim.groupby(activities_subset["assay_chembl_id"]).mean()


def assign_splits(
    activities_df: pd.DataFrame,
    compounds_df: pd.DataFrame,
    year_val_start: int,
    year_test_start: int,
    test_bins: list[SimilarityBin],
    split_val_like_test: bool = True,
) -> pd.DataFrame:
    """Assign a split label to each activity row based on doc_year and similarity bin.

    Binning is performed at the **assay level**: each assay is assigned to the bin
    matching the mean of its compounds' max Tanimoto similarities to the reference set.
    All compounds in an assay receive the same split label.

    Split logic:
    - doc_year < year_val_start                           → "train"
    - doc_year is null                                    → None
    - year_val_start <= doc_year < year_test_start:
        split_val_like_test=True  → "val_{bin_label}" per matching bin; unmatched → None
        split_val_like_test=False → "val"
    - doc_year >= year_test_start                         → "test_{bin_label}" per matching bin; unmatched → None

    Bin labels are derived from the SimilarityBin definitions, e.g.:
    - SimilarityBin(low=0.0, hi=0.35) → "sim_0.00_0.35"
    - SimilarityBin(equal=1.0)        → "sim_1.00"

    Similarity is looked up from compounds_df columns:
    - max_sim_pre_{year_test_start} for test rows
    - max_sim_pre_{year_val_start} for val rows (only when split_val_like_test=True)

    Args:
        activities_df: Activity data with at least columns:
            - ligand_chembl_id
            - assay_chembl_id
            - doc_year (numeric, nullable)
        compounds_df: DataFrame indexed by chembl_id with columns:
            - max_sim_pre_{year_test_start} (float)
            - max_sim_pre_{year_val_start} (float, required when split_val_like_test=True)
        year_val_start: First doc_year included in val; everything earlier is train.
        year_test_start: First doc_year included in test; must be > year_val_start.
        test_bins: Similarity bins defining the test (and optionally val) split labels.
        split_val_like_test: If True, apply the same bins to val rows. If False, all val
            rows get the single label "val".

    Returns:
        activities_df with a "split" column added (str, nullable).
    """
    result = activities_df.copy()
    year = result["doc_year"]
    split = pd.Series(index=result.index, dtype=object)

    split[year < year_val_start] = "train"

    mask_val = (year >= year_val_start) & (year < year_test_start)
    if mask_val.any():
        if split_val_like_test:
            val_subset = result[mask_val]
            assay_sim_val = _assay_mean_sim(
                val_subset, compounds_df[f"max_sim_pre_{year_val_start}"]
            )
            # Map assay-level mean sim back to individual activity rows
            row_sim_val = val_subset["assay_chembl_id"].map(assay_sim_val)
            for b in test_bins:
                bin_mask = mask_val.copy()
                bin_mask[mask_val] = _sim_matches_bin(row_sim_val, b).values
                split[bin_mask] = f"val_{_bin_label(b)}"
        else:
            split[mask_val] = "val"

    mask_test = year >= year_test_start
    if mask_test.any():
        test_subset = result[mask_test]
        assay_sim_test = _assay_mean_sim(
            test_subset, compounds_df[f"max_sim_pre_{year_test_start}"]
        )
        row_sim_test = test_subset["assay_chembl_id"].map(assay_sim_test)
        for b in test_bins:
            bin_mask = mask_test.copy()
            bin_mask[mask_test] = _sim_matches_bin(row_sim_test, b).values
            split[bin_mask] = f"test_{_bin_label(b)}"

    split = split.where(split.notna(), other=None)
    result["split"] = split
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the timesplit affinity benchmark."
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config.out_dir)
    intermediate_dir = out_dir / "intermediate"
    out_dir.mkdir(exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: ChEMBL queries
    # ------------------------------------------------------------------
    print("Step 1: Querying ChEMBL...")
    cr = config.chembl_requester
    requester = ChEMBLRequester(
        host=cr.host, user=cr.user, password=cr.password, dbname=cr.dbname
    )

    activities_df = pd.DataFrame(requester.get_all_single_protein_activity_data())
    print(f"  Activities raw: {activities_df.shape}")

    if config.pipeline.activity_limit is not None:
        # Use this when debugging to make runs faster
        n = min(config.pipeline.activity_limit, len(activities_df))
        activities_df = activities_df.sample(n=n, random_state=42).reset_index(
            drop=True
        )
        print(f"  Activities after limit: {activities_df.shape}")

    compounds_df = pd.DataFrame(requester.get_chembl_id_to_smiles())
    targets_df = pd.DataFrame(requester.get_single_protein_targets())
    assay_docs_df = pd.DataFrame(requester.get_assay_docs())

    # Filter compounds to only those referenced by the (possibly limited) activities
    relevant_ids = set(activities_df["ligand_chembl_id"].dropna())
    compounds_df = compounds_df[
        compounds_df["chembl_id"].isin(relevant_ids)
    ].reset_index(drop=True)

    print(f"  Compounds (filtered to activities): {compounds_df.shape}")

    # Drop compounds with no valid SMILES (null within compounds_df, or absent from
    # compound_structures entirely) and remove their activities.
    compounds_df = compounds_df[compounds_df["canonical_smiles"].notna()].reset_index(
        drop=True
    )
    valid_ids = set(compounds_df["chembl_id"])
    n_before_acts = len(activities_df)
    activities_df = activities_df[
        activities_df["ligand_chembl_id"].isin(valid_ids)
    ].reset_index(drop=True)
    n_dropped_acts = n_before_acts - len(activities_df)
    if n_dropped_acts:
        print(
            f"  Dropped {n_dropped_acts} activities with no valid SMILES ({len(activities_df)} remaining)"
        )
    print(f"  Targets: {targets_df.shape}")

    activities_df.to_parquet(intermediate_dir / "activities_raw.parquet", index=False)
    compounds_df.to_parquet(intermediate_dir / "compounds_raw.parquet", index=False)
    targets_df.to_parquet(intermediate_dir / "targets_raw.parquet", index=False)
    assay_docs_df.to_parquet(intermediate_dir / "assay_docs.parquet", index=False)
    print(
        "  Saved activities_raw.parquet, compounds_raw.parquet, targets_raw.parquet, assay_docs.parquet."
    )

    # ------------------------------------------------------------------
    # STEP 2: Compute fingerprints
    # ------------------------------------------------------------------
    print("Step 2: Computing fingerprints...")
    fp_names, fp_matrix = compute_ecfp4_fingerprints(
        mol_names=compounds_df["chembl_id"].tolist(),
        smiles=compounds_df["canonical_smiles"].tolist(),
        n_jobs=config.pipeline.n_jobs,
    )
    skipped = len(compounds_df) - len(fp_names)
    print(
        f"  Fingerprints: {fp_matrix.shape} ({skipped} molecules skipped due to parse failure)"
    )
    np.savez_compressed(
        intermediate_dir / "fingerprints.npz", names=fp_names, fps=fp_matrix
    )
    print("  Saved fingerprints.npz.")

    fp_index = {name: i for i, name in enumerate(fp_names)}

    # ------------------------------------------------------------------
    # STEP 3: Compute similarity
    # ------------------------------------------------------------------
    print("Step 3: Computing similarity...")

    year_test = config.pipeline.year_test_start
    year_val = config.pipeline.year_val_start

    sim_test = compute_similarity_for_cutoff_year(
        compounds_df=compounds_df,
        cutoff_year=year_test,
        fp_index=fp_index,
        fp_matrix=fp_matrix,
        n_jobs=config.pipeline.n_jobs,
    )
    sim_val = compute_similarity_for_cutoff_year(
        compounds_df=compounds_df,
        cutoff_year=year_val,
        fp_index=fp_index,
        fp_matrix=fp_matrix,
        n_jobs=config.pipeline.n_jobs,
    )

    print(
        f"  {year_test} cutoff (test) — sim range: [{sim_test[f'max_sim_pre_{year_test}'].min():.3f}, {sim_test[f'max_sim_pre_{year_test}'].max():.3f}]"
    )
    print(
        f"  {year_val} cutoff (val) — sim range: [{sim_val[f'max_sim_pre_{year_val}'].min():.3f}, {sim_val[f'max_sim_pre_{year_val}'].max():.3f}]"
    )

    # Add similarity columns to compounds_df and save as intermediate
    compounds_df = compounds_df.set_index("chembl_id")
    compounds_df = compounds_df.join(sim_test).join(sim_val)
    compounds_df.to_parquet(intermediate_dir / "compounds_with_novelty.parquet")
    print("  Saved compounds_with_novelty.parquet.")

    # ------------------------------------------------------------------
    # STEP 4: Extend pchembl_value columns to include values with relation != "="
    # ------------------------------------------------------------------
    print("Step 4: Adding pChEMBL columns...")
    n_original = activities_df["pchembl_value"].notna().sum()
    activities_df = add_pchembl_columns(activities_df)
    n_filled = activities_df["pchembl_value_filled"].notna().sum()
    print(
        f"  pchembl_value_filled: {n_filled} non-null (vs {n_original} in original pchembl_value)"
    )

    # Drop rows with no pchembl value or no relation (e.g. missing data or non-convertible units)
    mask_no_value = (
        activities_df["pchembl_value_filled"].isna()
        | activities_df["pchembl_relation"].isna()
    )
    n_dropped = mask_no_value.sum()
    if n_dropped:
        activities_df = activities_df[~mask_no_value].reset_index(drop=True)
        print(
            f"  Dropped {n_dropped} rows with null pchembl_value_filled or null pchembl_relation"
        )

    # ------------------------------------------------------------------
    # STEP 5: Assign splits
    # ------------------------------------------------------------------
    print("Step 5: Assigning splits...")
    activities_df = assign_splits(
        activities_df,
        compounds_df,
        year_val_start=year_val,
        year_test_start=year_test,
        test_bins=config.pipeline.test_set_similarity_bins,
        split_val_like_test=config.pipeline.split_val_like_test,
    )
    print("  Split value counts (including NaN):")
    print(activities_df["split"].value_counts(dropna=False).to_string())

    activities_df.to_parquet(
        intermediate_dir / "split_assignments.parquet", index=False
    )
    print("  Saved split_assignments.parquet.")

    # ------------------------------------------------------------------
    # STEP 6: Filter assay-types in val/test splits (optional)
    # ------------------------------------------------------------------
    af = config.pipeline.filter_val_and_test_sets
    if af is not None:
        print("Step 6: Filtering assay-types in val/test splits...")
        before = len(activities_df)
        activities_df = filter_assay_types(
            activities_df=activities_df,
            assay_docs_df=assay_docs_df,
            apply_to=af.apply_to,
            only_equal_relation=af.only_equal_relation,
            min_cpd_per_assay=af.min_cpd_per_assay,
            min_std=af.min_std,
            one_assay_per_doc=af.one_assay_per_doc,
        )
        print(
            f"  Rows removed: {before - len(activities_df):,}  ({before:,} → {len(activities_df):,})"
        )
        print("  Split distribution after filtering:")
        print(activities_df["split"].value_counts(dropna=False).to_string())
    else:
        print("Step 6: Assay filtering disabled (filter_val_and_test_sets is null).")

    # ------------------------------------------------------------------
    # STEP 7: Build final activity file
    # ------------------------------------------------------------------
    print("Step 7: Building final activity file...")

    sim_cols = compounds_df[
        [
            "cpd_earliest_year",
            "canonical_smiles",
            "mw_freebase",
            f"max_sim_pre_{year_test}",
            f"most_sim_cpd_pre_{year_test}",
            f"max_sim_pre_{year_val}",
            f"most_sim_cpd_pre_{year_val}",
        ]
    ]
    activities_df = activities_df.merge(
        sim_cols,
        left_on="ligand_chembl_id",
        right_index=True,
        how="left",
    )

    activities_df = activities_df[activities_df["split"].notna()]

    final_col_order = [
        "target_chembl_id",
        "assay_chembl_id",
        "ligand_chembl_id",
        "standard_type",
        "pchembl_relation",
        "pchembl_value_filled",
        "split",
        "mw_freebase",
        "data_validity_comment",
        "potential_duplicate",
        "doc_year",
        "cpd_earliest_year",
        f"max_sim_pre_{year_test}",
        f"most_sim_cpd_pre_{year_test}",
        f"max_sim_pre_{year_val}",
        f"most_sim_cpd_pre_{year_val}",
        "canonical_smiles",
    ]
    activities_df = activities_df[final_col_order]

    activities_df.to_parquet(out_dir / "activities.parquet", index=False)
    print(f"  Saved activities.parquet: {activities_df.shape}")

    # ------------------------------------------------------------------
    # STEP 8: Save target file
    # ------------------------------------------------------------------
    print("Step 8: Saving target file...")
    targets_df.to_parquet(out_dir / "targets.parquet", index=False)
    print(f"  Saved targets.parquet: {targets_df.shape}")

    print("\nFinal split distribution:")
    print(activities_df["split"].value_counts(dropna=False).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
