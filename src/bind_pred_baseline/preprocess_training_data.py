"""Preprocess activities data into train/val/test split .npz files for the baseline.

Input (paths specified in config):
  - activities.parquet   (activity rows with pchembl values)
  - targets.parquet      (target metadata including uniprot_id and sequence)

Output (written to paths.output_dir):
  - fingerprints_ecfp4.npz  (ECFP4 fingerprints for all compounds)
  - mol_properties.npz      (normalised physicochemical properties)
  - train.npz / val.npz / test.npz
                            (fp_indices, target_indices, standard_type_indices, labels, assay_ids, ligand_ids)
  - target_index.json       (uniprot_id → integer index mapping)
  - meta.json               (n_targets, n_standard_types, fp_size, fp_type)
  - oov_target_mapping.json (OOV uniprot_id → nearest training target; only written if OOV targets exist)

OOV targets (not seen in train) are auto-mapped to the most sequence-similar
training target via BLOSUM62 alignment.

Usage:
    uv run python -m bind_pred_baseline.preprocess_training_data \\
        --config configs/baseline/data.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bind_pred_baseline.constants import STANDARD_TYPE_INDEX
from bind_pred_baseline.preprocess_utils import (
    FEATURE_NAMES as PROP_FEATURE_NAMES,
    average_duplicates,
    build_target_index,
    compute_fingerprints,
    compute_mol_properties,
    find_closest_training_targets,
    load_activities,
    load_split_from_file,
    normalise_mol_properties,
    resolve_target_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess data for the binding prediction baseline."
    )
    parser.add_argument("--config", required=True, help="Path to data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    activities_path = Path(cfg["paths"]["activities_parquet"])
    targets_path = Path(cfg["paths"]["targets_parquet"])
    out_dir = Path(cfg["paths"]["output_dir"])
    val_splits = cfg.get("val_splits", ["val_novel", "val_not_novel"])
    n_jobs: int = cfg.get("n_jobs", 1)
    split_from_file: str | None = cfg.get("split_from_file", None)
    min_train_datapoints: int = cfg.get("oov_min_train_datapoints", 0)
    fp_type: str = cfg.get("fp_type", "binary")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and filter data; join uniprot_id from targets
    # ------------------------------------------------------------------
    print("\nLoading and filtering activities...")
    df = load_activities(activities_path)
    print(f"  {len(df)} IC50/Ki/Kd '=' rows")

    targets_df = pd.read_parquet(targets_path)[
        ["target_chembl_id", "uniprot_id", "target_name", "sequence"]
    ]
    df = df.merge(targets_df, on="target_chembl_id", how="left")
    n_no_uniprot = df["uniprot_id"].isna().sum()
    if n_no_uniprot:
        missing_targets = sorted(
            df[df["uniprot_id"].isna()]["target_chembl_id"].unique().tolist()
        )
        raise RuntimeError(
            f"{n_no_uniprot} activity rows have no uniprot_id after joining targets.parquet. "
            f"Affected target_chembl_ids: {missing_targets}"
        )
    print(f"  {df['uniprot_id'].nunique()} unique uniprot_ids across all splits")

    if split_from_file:
        print(f"\nLoading split assignments from {split_from_file}...")
        split_map = load_split_from_file(split_from_file)
        n_before = len(df)
        df = df[df["uniprot_id"].isin(split_map)].copy()
        n_dropped = n_before - len(df)
        print(
            f"  Dropped {n_dropped} rows whose uniprot_id was not in the split file → {len(df)} remaining"
        )
        df["split"] = df["uniprot_id"].map(split_map)
        print(
            f"  Split counts: { {s: int((df['split'] == s).sum()) for s in sorted(df['split'].unique())} }"
        )

    # ------------------------------------------------------------------
    # Generate fingerprints from compounds
    # ------------------------------------------------------------------
    fp_npz_path = out_dir / "fingerprints_ecfp4.npz"
    print(f"\nGenerating {fp_type} ECFP4 fingerprints...")
    compounds_df = (
        df[["ligand_chembl_id", "canonical_smiles"]]
        .drop_duplicates(subset="ligand_chembl_id")
        .rename(columns={"ligand_chembl_id": "chembl_id"})
        .reset_index(drop=True)
    )
    fp_names, fp_matrix = compute_fingerprints(
        mol_names=compounds_df["chembl_id"].tolist(),
        smiles=compounds_df["canonical_smiles"].tolist(),
        fp_type=fp_type,
        n_jobs=n_jobs,
    )
    skipped = len(compounds_df) - len(fp_names)
    print(
        f"  Fingerprints: {fp_matrix.shape} ({skipped} molecules skipped due to parse failure)"
    )

    fp_size: int = fp_matrix.shape[1]

    # ------------------------------------------------------------------
    # Compute raw mol properties & normalise
    # ------------------------------------------------------------------
    print(f"\nComputing mol properties for {len(compounds_df)} compounds...")
    prop_names, raw_props = compute_mol_properties(
        mol_names=compounds_df["chembl_id"].tolist(),
        smiles=compounds_df["canonical_smiles"].tolist(),
        n_jobs=n_jobs,
    )
    skipped_props = len(compounds_df) - len(prop_names)
    print(
        f"  Mol properties: {raw_props.shape} ({skipped_props} molecules skipped due to parse failure)"
    )

    print("\nNormalising mol properties (fit on training compounds only)...")
    train_compound_ids = set(df[df["split"] == "train"]["ligand_chembl_id"].unique())
    train_prop_indices = np.array(
        [i for i, name in enumerate(prop_names) if name in train_compound_ids],
        dtype=np.int64,
    )
    if len(train_prop_indices) == 0:
        raise RuntimeError(
            "No training compounds found for normalisation — check your split column."
        )
    train_mean = raw_props[train_prop_indices].mean(axis=0).astype(np.float32)
    train_std = raw_props[train_prop_indices].std(axis=0).astype(np.float32)
    normed_props = normalise_mol_properties(raw_props, mean=train_mean, std=train_std)
    print(
        f"  Normalised using {len(train_prop_indices)} training compounds to fit scaler"
    )

    # ------------------------------------------------------------------
    # Jointly filter to compounds with both fingerprint and mol properties,
    # then save both NPZ files
    # ------------------------------------------------------------------
    print("\nJointly filtering compound representations...")
    prop_name_to_row: dict[str, int] = {name: i for i, name in enumerate(prop_names)}
    common = set(fp_names) & prop_name_to_row.keys()
    n_fp_only = len(fp_names) - len(common)
    n_prop_only = len(prop_names) - len(common)
    if n_fp_only:
        print(f"  Dropped {n_fp_only} compounds with fingerprint but no mol properties")
    if n_prop_only:
        print(
            f"  Dropped {n_prop_only} compounds with mol properties but no fingerprint"
        )
    print(f"  {len(common)} compounds retained with both representations")

    keep_mask = np.array([n in common for n in fp_names])
    fp_names = fp_names[keep_mask]
    fp_matrix = fp_matrix[keep_mask]
    normed_props = normed_props[[prop_name_to_row[n] for n in fp_names]]

    np.savez_compressed(fp_npz_path, names=fp_names, fps=fp_matrix)
    print(f"  Saved {fp_npz_path}")

    prop_npz_path = out_dir / "mol_properties.npz"
    np.savez_compressed(
        prop_npz_path,
        names=fp_names,
        props=normed_props,
        feature_names=np.array(PROP_FEATURE_NAMES),
        mean=train_mean,
        std=train_std,
    )
    print(f"  Saved {prop_npz_path}")

    fp_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(fp_names)}

    n_before = len(df)
    df = df[df["ligand_chembl_id"].isin(fp_name_to_idx)].copy()
    print(
        f"  Dropped {n_before - len(df)} activity rows with no compound representation → {len(df)} remaining"
    )

    # ------------------------------------------------------------------
    # Build target index from train uniprot_ids and map out-of-vocabulary (OOV)
    # targets to the uniprot_id with the most similar protein sequence
    # ------------------------------------------------------------------
    train_df = df[df["split"] == "train"]
    index = build_target_index(train_df["uniprot_id"].tolist())
    train_counts: dict[str, int] = train_df["uniprot_id"].value_counts().to_dict()
    print(f"\n  {len(index)} unique training uniprot_ids")

    with open(out_dir / "target_index.json", "w") as f:
        json.dump(index, f)
    print("  Saved target_index.json")

    non_train = df[df["split"] != "train"]
    oov = sorted({uid for uid in non_train["uniprot_id"].unique() if uid not in index})

    oov_mapping: dict[str, str] = {}
    if oov:
        print(
            f"\nAuto-mapping {len(oov)} OOV uniprot_id(s) by sequence similarity (BLOSUM62)..."
        )
        sequences: dict[str, str | None] = dict(
            zip(
                targets_df["uniprot_id"],
                targets_df.get("sequence", [None] * len(targets_df)),
            )
        )
        oov_mapping = find_closest_training_targets(
            oov,
            list(index.keys()),
            sequences,
            n_jobs=n_jobs,
            train_counts=train_counts,
            min_train_datapoints=min_train_datapoints,
        )

        with open(out_dir / "oov_target_mapping.json", "w") as f:
            json.dump(oov_mapping, f, indent=2)
        print("  Saved oov_target_mapping.json")

    # ------------------------------------------------------------------
    # Post processing and saving
    # ------------------------------------------------------------------

    # Composite assay ID includes standard_type so that Pearson r is computed
    # within a single measurement type (an assay_chembl_id can have IC50, Ki, and Kd rows).
    df["assay_id_for_eval"] = df["assay_chembl_id"] + "_" + df["standard_type"]
    df["standard_type_idx"] = df["standard_type"].map(STANDARD_TYPE_INDEX)

    split_defs = {
        "train": df[df["split"] == "train"].copy(),
        "val": df[df["split"].isin(val_splits)].copy(),
        "test": df[df["split"] == "test"].copy(),
    }

    for split_name, split_df in split_defs.items():
        print(f"\nProcessing {split_name} ({len(split_df)} rows)...")

        if split_name == "train":
            split_df = average_duplicates(split_df)
            print(f"  After averaging duplicates: {len(split_df)} rows")

        target_indices = resolve_target_ids(
            split_df["uniprot_id"].tolist(), index, mapping=oov_mapping or None
        )
        fp_indices = [fp_name_to_idx[lid] for lid in split_df["ligand_chembl_id"]]

        out_path = out_dir / f"{split_name}.npz"
        np.savez_compressed(
            out_path,
            fp_indices=np.array(fp_indices, dtype=np.int64),
            target_indices=np.array(target_indices, dtype=np.int64),
            standard_type_indices=np.array(
                split_df["standard_type_idx"].tolist(), dtype=np.int64
            ),
            labels=split_df["pchembl_value_filled"].values.astype(np.float32),
            assay_ids=np.array(split_df["assay_id_for_eval"].tolist()),
            ligand_ids=np.array(split_df["ligand_chembl_id"].tolist()),
        )
        print(f"  Saved {out_path.name}: {len(fp_indices)} samples")

    meta = {
        "n_targets": len(index),
        "n_standard_types": len(STANDARD_TYPE_INDEX),
        "fp_size": fp_size,
        "fp_type": fp_type,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    print(f"\nSaved meta.json: {meta}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
