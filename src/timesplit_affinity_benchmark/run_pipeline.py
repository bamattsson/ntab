"""
Pipeline entry point for generating the timesplit affinity benchmark.

Usage:
    python src/timesplit_affinity_benchmark/run_pipeline.py --config config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from timesplit_affinity_benchmark.affinity_utils import add_pchembl_columns
from timesplit_affinity_benchmark.assay_filter import filter_assay_types
from timesplit_affinity_benchmark.chembl_requester import ChEMBLRequester
from timesplit_affinity_benchmark.config import load_config
from timesplit_affinity_benchmark.mol_fingerprints import compute_ecfp4_fingerprints
from timesplit_affinity_benchmark.novelty import compute_novelty_for_cutoff
from timesplit_affinity_benchmark.split_assigner import assign_splits


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
    # STEP 3: Compute novelty
    # ------------------------------------------------------------------
    print("Step 3: Computing novelty...")

    year_test = config.pipeline.year_test_start
    year_val = config.pipeline.year_val_start

    novelty_test = compute_novelty_for_cutoff(
        compounds_df=compounds_df,
        cutoff_year=year_test,
        fp_index=fp_index,
        fp_matrix=fp_matrix,
        threshold=config.pipeline.tanimoto_threshold,
        n_jobs=config.pipeline.n_jobs,
    )
    novel_test = novelty_test[f"is_novel_{year_test}"]
    print(
        f"  {year_test} cutoff (test) — novel: {(novel_test == True).sum()}, not novel: {(novel_test == False).sum()}, reference: {novel_test.isna().sum()}"
    )

    novelty_val = compute_novelty_for_cutoff(
        compounds_df=compounds_df,
        cutoff_year=year_val,
        fp_index=fp_index,
        fp_matrix=fp_matrix,
        threshold=config.pipeline.tanimoto_threshold,
        n_jobs=config.pipeline.n_jobs,
    )
    novel_val = novelty_val[f"is_novel_{year_val}"]
    print(
        f"  {year_val} cutoff (val) — novel: {(novel_val == True).sum()}, not novel: {(novel_val == False).sum()}, reference: {novel_val.isna().sum()}"
    )

    # Add 6 novelty columns to compounds_df and save as intermediate
    compounds_df = compounds_df.set_index("chembl_id")
    compounds_df = compounds_df.join(novelty_test).join(novelty_val)
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
            one_assay_per_doi=af.one_assay_per_doi,
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
