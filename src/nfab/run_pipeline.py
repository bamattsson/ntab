"""
Pipeline entry point for generating the timesplit affinity benchmark.

Usage:
    python src/nfab/run_pipeline.py --config config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfab.affinity_utils import add_pchembl_columns
from nfab.assay_filter import filter_assay_types
from nfab.chembl_requester import ChEMBLRequester
from nfab.config import load_config
from nfab.similarity import (
    compute_ecfp4_fingerprints,
    compute_similarity_for_cutoff_year,
)


def assign_splits(
    activities_df: pd.DataFrame,
    compounds_df: pd.DataFrame,
    year_val_start: int = 2022,
    year_test_start: int = 2023,
) -> pd.DataFrame:
    """Assign a split label to each activity row based on doc_year and novelty.

    Split logic (with default year_val_start=2022, year_test_start=2023):
    - doc_year < year_val_start                                                    → "train"
    - year_val_start <= doc_year < year_test_start and is_novel == True            → "val_novel"
    - year_val_start <= doc_year < year_test_start and is_novel != True (or NaN)   → "val_not_novel"
    - doc_year >= year_test_start and is_novel == True                             → "test"
    - doc_year >= year_test_start and is_novel != True (or NaN)                   → "discard_not_novel"
    - doc_year is null                                                              → None

    Novelty is read from compounds_df columns named ``is_novel_{year_val_start}``
    and ``is_novel_{year_test_start}``, computed in run_pipeline from
    ``compute_similarity_for_cutoff_year`` output via ``max_sim < threshold``.
    Reference-set compounds (cpd_earliest_year < cutoff) have max_sim=1.0 and
    are therefore treated as not novel for any threshold < 1.0.

    Args:
        activities_df: Activity data with at least columns:
            - ligand_chembl_id
            - doc_year (numeric, nullable)
        compounds_df: DataFrame indexed by chembl_id with columns:
            - is_novel_{year_val_start} (bool or pd.NA)
            - is_novel_{year_test_start} (bool or pd.NA)
            Derived from max_sim_pre_{year} < threshold in run_pipeline.
        year_val_start: First doc_year included in val; everything earlier is train.
        year_test_start: First doc_year included in test; must be > year_val_start.

    Returns:
        activities_df with a "split" column added (str, nullable).
    """
    result = activities_df.copy()
    year = result["doc_year"]
    split = pd.Series(index=result.index, dtype=object)

    split[year < year_val_start] = "train"

    mask_val = (year >= year_val_start) & (year < year_test_start)
    if mask_val.any():
        # .eq(True) returns a proper bool series: True → True, False/NA → False
        is_novel_val = (
            result["ligand_chembl_id"]
            .map(compounds_df[f"is_novel_{year_val_start}"])
            .eq(True)
        )
        split[mask_val & is_novel_val] = "val_novel"
        split[mask_val & ~is_novel_val] = "val_not_novel"

    mask_test = year >= year_test_start
    if mask_test.any():
        is_novel_test = (
            result["ligand_chembl_id"]
            .map(compounds_df[f"is_novel_{year_test_start}"])
            .eq(True)
        )
        split[mask_test & is_novel_test] = "test"
        split[mask_test & ~is_novel_test] = "discard_not_novel"

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
    threshold = config.pipeline.tanimoto_threshold

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

    # Derive is_novel from similarity; when threshold is None all compounds are novel
    for year, sim_df in [(year_test, sim_test), (year_val, sim_val)]:
        if threshold is not None:
            sim_df[f"is_novel_{year}"] = sim_df[f"max_sim_pre_{year}"] < threshold
        else:
            sim_df[f"is_novel_{year}"] = True

    novel_test = sim_test[f"is_novel_{year_test}"]
    print(
        f"  {year_test} cutoff (test) — novel: {(novel_test == True).sum()}, not novel: {(novel_test == False).sum()}"
    )
    novel_val = sim_val[f"is_novel_{year_val}"]
    print(
        f"  {year_val} cutoff (val) — novel: {(novel_val == True).sum()}, not novel: {(novel_val == False).sum()}"
    )

    # Add similarity and novelty columns to compounds_df and save as intermediate
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
    if not config.pipeline.keep_discard_not_novel:
        activities_df = activities_df[activities_df["split"] != "discard_not_novel"]

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
