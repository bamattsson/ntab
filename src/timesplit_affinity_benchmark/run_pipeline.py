"""
Pipeline entry point for generating the timesplit affinity benchmark.

Usage:
    python src/timesplit_affinity_benchmark/run_pipeline.py --config config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from timesplit_affinity_benchmark.chembl_requester import ChEMBLRequester
from timesplit_affinity_benchmark.config import load_config
from timesplit_affinity_benchmark.mol_fingerprints import compute_ecfp4_fingerprints
from timesplit_affinity_benchmark.split_assigner import assign_splits
from timesplit_affinity_benchmark.tanimoto_filter import filter_by_tanimoto

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
        activities_df = activities_df.head(config.pipeline.activity_limit)
        print(f"  Activities after limit: {activities_df.shape}")

    compounds_df = pd.DataFrame(requester.get_chembl_id_to_smiles())
    if config.pipeline.activity_limit is not None:
        relevant_ids = set(activities_df["ligand_chembl_id"])
        compounds_df = compounds_df[compounds_df["chembl_id"].isin(relevant_ids)]
    print(f"  Compounds: {compounds_df.shape}")

    targets_df = pd.DataFrame(requester.get_single_protein_targets())
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

    # ------------------------------------------------------------------
    # STEP 3: Assign splits
    # ------------------------------------------------------------------
    print("Step 3: Assigning splits...")
    fp_index = {name: i for i, name in enumerate(fp_names)}

    # Reference set: unique train+val compounds (doc_year < 2024)
    trainval_ids = (
        activities_df[activities_df["doc_year"] < 2024]["ligand_chembl_id"]
        .dropna().unique()
    )
    ref_ids = np.array([cid for cid in trainval_ids if cid in fp_index])
    ref_fps = fp_matrix[[fp_index[cid] for cid in ref_ids]]
    print(f"  Reference compounds (train+val, deduplicated): {len(ref_ids)}")

    # Candidate set: unique 2024+ compounds
    candidate_ids_all = (
        activities_df[activities_df["doc_year"] >= 2024]["ligand_chembl_id"]
        .dropna().unique()
    )
    cand_ids = np.array([cid for cid in candidate_ids_all if cid in fp_index])
    cand_fps = fp_matrix[[fp_index[cid] for cid in cand_ids]]
    print(f"  Candidate compounds (2024+, deduplicated): {len(cand_ids)}")

    is_novel, max_sims, most_similar_ids = filter_by_tanimoto(
        candidate_fps=cand_fps,
        reference_fps=ref_fps,
        reference_ids=ref_ids,
        threshold=config.pipeline.tanimoto_threshold,
        n_jobs=config.pipeline.n_jobs,
    )

    is_novel_df = pd.DataFrame(
        {
            "is_novel": is_novel,
            "max_similarity": max_sims,
            "most_similar_id": most_similar_ids,
        },
        index=pd.Index(cand_ids, name="chembl_id"),
    )
    is_novel_df.to_parquet(INTERMEDIATE_DIR / "is_novel.parquet")
    print(f"  Saved is_novel.parquet. Novel: {is_novel.sum()} / {len(is_novel)}")

    activities_df = assign_splits(activities_df, is_novel_df)
    print("  Split value counts (including NaN):")
    print(activities_df["split"].value_counts(dropna=False).to_string())

    activities_df.to_parquet(INTERMEDIATE_DIR / "split_assignments.parquet", index=False)
    print("  Saved split_assignments.parquet.")

    # ------------------------------------------------------------------
    # STEP 4: Build final activity file
    # ------------------------------------------------------------------
    print("Step 4: Building final activity file...")
    compounds_lookup = compounds_df.set_index("chembl_id")[["canonical_smiles", "earliest_year"]]
    activities_df = activities_df.merge(
        compounds_lookup, left_on="ligand_chembl_id", right_index=True, how="left"
    )
    activities_df = activities_df.merge(
        is_novel_df[["most_similar_id", "max_similarity"]],
        left_on="ligand_chembl_id", right_index=True, how="left",
    )

    activities_df = activities_df[activities_df["split"].notna()]
    if not config.pipeline.keep_not_novel:
        activities_df = activities_df[activities_df["split"] != "2024_not_novel"]

    activities_df.to_parquet(OUT_DIR / "activities.parquet", index=False)
    print(f"  Saved activities.parquet: {activities_df.shape}")

    # ------------------------------------------------------------------
    # STEP 5: Save target file
    # ------------------------------------------------------------------
    print("Step 5: Saving target file...")
    targets_df.to_parquet(OUT_DIR / "targets.parquet", index=False)
    print(f"  Saved targets.parquet: {targets_df.shape}")

    print("\nFinal split distribution:")
    print(activities_df["split"].value_counts(dropna=False).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
