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
from timesplit_affinity_benchmark.chembl_requester import ChEMBLRequester
from timesplit_affinity_benchmark.config import load_config
from timesplit_affinity_benchmark.mol_fingerprints import compute_ecfp4_fingerprints
from timesplit_affinity_benchmark.novelty import compute_novelty_for_cutoff
from timesplit_affinity_benchmark.split_assigner import assign_splits

INTERMEDIATE_DIR = Path("intermediate_out")
OUT_DIR = Path("out")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the timesplit affinity benchmark.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: ChEMBL queries
    # ------------------------------------------------------------------
    print("Step 1: Querying ChEMBL...")
    cr = config.chembl_requester
    requester = ChEMBLRequester(host=cr.host, user=cr.user, password=cr.password, dbname=cr.dbname)

    activities_df = pd.DataFrame(requester.get_all_single_protein_activity_data())
    print(f"  Activities raw: {activities_df.shape}")

    if config.pipeline.activity_limit is not None:
        n = min(config.pipeline.activity_limit, len(activities_df))
        activities_df = activities_df.sample(n=n, random_state=42).reset_index(drop=True)
        print(f"  Activities after limit: {activities_df.shape}")

    compounds_df = pd.DataFrame(requester.get_chembl_id_to_smiles())
    targets_df = pd.DataFrame(requester.get_single_protein_targets())

    # Filter compounds to only those referenced by the (possibly limited) activities
    relevant_ids = set(activities_df["ligand_chembl_id"].dropna())
    compounds_df = compounds_df[compounds_df["chembl_id"].isin(relevant_ids)].reset_index(drop=True)

    print(f"  Compounds (filtered to activities): {compounds_df.shape}")
    print(f"  Targets: {targets_df.shape}")

    activities_df.to_parquet(INTERMEDIATE_DIR / "activities_raw.parquet", index=False)
    compounds_df.to_parquet(INTERMEDIATE_DIR / "compounds_raw.parquet", index=False)
    targets_df.to_parquet(INTERMEDIATE_DIR / "targets_raw.parquet", index=False)
    print("  Saved activities_raw.parquet, compounds_raw.parquet, targets_raw.parquet.")

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
    print(f"  Fingerprints: {fp_matrix.shape} ({skipped} molecules skipped due to parse failure)")
    np.savez_compressed(INTERMEDIATE_DIR / "fingerprints.npz", names=fp_names, fps=fp_matrix)
    print("  Saved fingerprints.npz.")

    fp_index = {name: i for i, name in enumerate(fp_names)}

    # ------------------------------------------------------------------
    # STEP 3: Compute novelty
    # ------------------------------------------------------------------
    print("Step 3: Computing novelty...")

    novelty_2024 = compute_novelty_for_cutoff(
        compounds_df=compounds_df,
        cutoff_year=2024,
        fp_index=fp_index,
        fp_matrix=fp_matrix,
        threshold=config.pipeline.tanimoto_threshold,
        n_jobs=config.pipeline.n_jobs,
    )
    novel_2024 = novelty_2024["is_novel_2024"]
    print(f"  2024 cutoff — novel: {(novel_2024 == True).sum()}, not novel: {(novel_2024 == False).sum()}, reference: {novel_2024.isna().sum()}")

    novelty_2023 = compute_novelty_for_cutoff(
        compounds_df=compounds_df,
        cutoff_year=2023,
        fp_index=fp_index,
        fp_matrix=fp_matrix,
        threshold=config.pipeline.tanimoto_threshold,
        n_jobs=config.pipeline.n_jobs,
    )
    novel_2023 = novelty_2023["is_novel_2023"]
    print(f"  2023 cutoff — novel: {(novel_2023 == True).sum()}, not novel: {(novel_2023 == False).sum()}, reference: {novel_2023.isna().sum()}")

    # Enrich compounds_df with all 6 novelty columns and save as intermediate
    compounds_df = compounds_df.set_index("chembl_id")
    compounds_df = compounds_df.join(novelty_2024).join(novelty_2023)
    compounds_df.to_parquet(INTERMEDIATE_DIR / "compounds_with_novelty.parquet")
    print("  Saved compounds_with_novelty.parquet.")

    # ------------------------------------------------------------------
    # STEP 4: Add pChEMBL columns
    # ------------------------------------------------------------------
    print("Step 4: Adding pChEMBL columns...")
    n_original = activities_df["pchembl_value"].notna().sum()
    activities_df = add_pchembl_columns(activities_df)
    n_filled = activities_df["pchembl_value_filled"].notna().sum()
    print(f"  pchembl_value_filled: {n_filled} non-null (vs {n_original} in original pchembl_value)")

    # ------------------------------------------------------------------
    # STEP 5: Assign splits
    # ------------------------------------------------------------------
    print("Step 5: Assigning splits...")
    activities_df = assign_splits(activities_df, compounds_df)
    print("  Split value counts (including NaN):")
    print(activities_df["split"].value_counts(dropna=False).to_string())

    activities_df.to_parquet(INTERMEDIATE_DIR / "split_assignments.parquet", index=False)
    print("  Saved split_assignments.parquet.")

    # ------------------------------------------------------------------
    # STEP 6: Build final activity file
    # ------------------------------------------------------------------
    print("Step 6: Building final activity file...")

    sim_cols = compounds_df[["cpd_earliest_year", "canonical_smiles", "mw_freebase",
                              "max_sim_pre_2024", "most_sim_cpd_pre_2024",
                              "max_sim_pre_2023", "most_sim_cpd_pre_2023"]]
    activities_df = activities_df.merge(
        sim_cols,
        left_on="ligand_chembl_id",
        right_index=True,
        how="left",
    )

    activities_df = activities_df[activities_df["split"].notna()]
    if not config.pipeline.keep_not_novel_in_test:
        activities_df = activities_df[activities_df["split"] != "2024_not_novel"]

    final_col_order = [
        "target_chembl_id", "assay_chembl_id", "ligand_chembl_id",
        "standard_type", "pchembl_relation", "pchembl_value_filled",
        "split", "mw_freebase", "mutation", "data_validity_comment", "potential_duplicate",
        "doc_year", "cpd_earliest_year",
        "max_sim_pre_2024", "most_sim_cpd_pre_2024",
        "max_sim_pre_2023", "most_sim_cpd_pre_2023",
        "canonical_smiles",
    ]
    activities_df = activities_df[final_col_order]

    activities_df.to_parquet(OUT_DIR / "activities.parquet", index=False)
    print(f"  Saved activities.parquet: {activities_df.shape}")

    # ------------------------------------------------------------------
    # STEP 7: Save target file
    # ------------------------------------------------------------------
    print("Step 7: Saving target file...")
    targets_df.to_parquet(OUT_DIR / "targets.parquet", index=False)
    print(f"  Saved targets.parquet: {targets_df.shape}")

    print("\nFinal split distribution:")
    print(activities_df["split"].value_counts(dropna=False).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
