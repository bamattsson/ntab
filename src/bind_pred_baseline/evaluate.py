"""Evaluation script: run inference on held-out splits from activities.parquet.

Computes fingerprints and molecular properties inline from SMILES, respects
per-row standard_type, and writes a per-row predictions CSV alongside the
measured labels. Also prints size-weighted Pearson r per split.

Usage
-----
    python -m bind_pred_baseline.evaluate \\
        --checkpoint out_baseline/lightning_logs/version_0/checkpoints/best.ckpt \\
        --data-dir out_baseline/data_preprocessing \\
        --activities out/activities.parquet \\
        --targets out/targets.parquet \\
        --splits test 2024_not_novel \\
        --output predictions.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bind_pred_baseline.constants import MIN_ASSAY_SIZE, MOL_PROP_FEATURES, STANDARD_TYPE_INDEX
from bind_pred_baseline.dataset import PredictDataset
from bind_pred_baseline.model import AffinityModel
from bind_pred_baseline.model_utils import pearson_r_per_assay
from bind_pred_baseline.preprocess_utils import (
    FEATURE_NAMES as PROP_FEATURE_NAMES,
    compute_fingerprints,
    compute_mol_properties,
    normalise_mol_properties,
    resolve_target_ids,
)

OUTPUT_COLUMNS = [
    "assay_chembl_id",
    "ligand_chembl_id",
    "uniprot_id",
    "standard_type",
    "split",
    "pchembl_value_filled",
    "pred_pchembl",
]


def preprocess_activities_for_eval(
    df: pd.DataFrame,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Preprocess an activities DataFrame for evaluation inference.

    Computes fingerprints and molecular properties inline from SMILES,
    normalises using training scalers, and resolves target and standard-type
    indices from training artifacts. Rows where SMILES fails to parse are
    dropped with a warning.

    Args:
        df: DataFrame with columns: ligand_chembl_id, canonical_smiles,
            uniprot_id, standard_type, assay_chembl_id, split,
            pchembl_value_filled (plus any additional columns, which are
            preserved in df_eval).
        data_dir: Training preprocessing directory containing target_index.json,
            meta.json, mol_properties.npz, and optionally oov_target_mapping.json.

    Returns:
        Tuple of:
            fp_matrix:        float32 array (N_unique_compounds, fp_size)
            props_matrix:     float32 array (N_unique_compounds, n_mol_prop_features)
            fp_indices:       int64 array   (N_rows,)
            target_indices:   int64 array   (N_rows,)
            std_type_indices: int64 array   (N_rows,) — per-row standard type
            df_eval:          DataFrame     (N_rows,) in the same row order as
                              the arrays above, with the same columns as df.
    """
    with open(data_dir / "target_index.json") as f:
        target_index: dict[str, int] = json.load(f)

    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    fp_size: int = meta["fp_size"]
    fp_type: str = meta["fp_type"]

    oov_mapping: dict[str, str] = {}
    oov_path = data_dir / "oov_target_mapping.json"
    if oov_path.exists():
        with open(oov_path) as f:
            oov_mapping = json.load(f)

    unresolvable = sorted(
        uid for uid in df["uniprot_id"].unique()
        if uid not in target_index and oov_mapping.get(uid, uid) not in target_index
    )
    if unresolvable:
        raise KeyError(
            f"Cannot resolve these uniprot_ids to any training target: {unresolvable}"
        )

    # Deduplicate by ligand_chembl_id for fingerprint/property computation
    unique_cpds = df[["ligand_chembl_id", "canonical_smiles"]].drop_duplicates("ligand_chembl_id")

    fp_names, fp_matrix_raw = compute_fingerprints(
        mol_names=unique_cpds["ligand_chembl_id"].tolist(),
        smiles=unique_cpds["canonical_smiles"].tolist(),
        fp_type=fp_type,
        fp_size=fp_size,
    )

    prop_names, raw_props = compute_mol_properties(
        mol_names=unique_cpds["ligand_chembl_id"].tolist(),
        smiles=unique_cpds["canonical_smiles"].tolist(),
    )

    prop_name_to_row: dict[str, int] = {name: i for i, name in enumerate(prop_names)}
    common = set(fp_names) & prop_name_to_row.keys()

    n_dropped = len(unique_cpds) - len(common)
    if n_dropped > 0:
        print(f"WARNING: {n_dropped} compound(s) dropped due to SMILES parse failure.")

    keep_mask = np.array([n in common for n in fp_names])
    fp_names_kept = fp_names[keep_mask]
    fp_matrix_kept = fp_matrix_raw[keep_mask].astype(np.float32)
    raw_props_kept = raw_props[[prop_name_to_row[n] for n in fp_names_kept]]

    train_props_npz = np.load(data_dir / "mol_properties.npz")
    normed_props = normalise_mol_properties(
        raw_props_kept,
        mean=train_props_npz["mean"],
        std=train_props_npz["std"],
    )

    feature_names = list(PROP_FEATURE_NAMES)
    col_indices = [feature_names.index(f) for f in MOL_PROP_FEATURES]
    props_matrix = normed_props[:, col_indices].astype(np.float32)

    df_eval = df[df["ligand_chembl_id"].isin(common)].reset_index(drop=True)

    fp_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(fp_names_kept)}

    fp_indices = np.array(
        [fp_name_to_idx[n] for n in df_eval["ligand_chembl_id"]], dtype=np.int64
    )
    target_indices = np.array(
        resolve_target_ids(
            df_eval["uniprot_id"].tolist(), target_index, mapping=oov_mapping or None
        ),
        dtype=np.int64,
    )
    std_type_indices = np.array(
        [STANDARD_TYPE_INDEX[st] for st in df_eval["standard_type"]], dtype=np.int64
    )

    return fp_matrix_kept, props_matrix, fp_indices, target_indices, std_type_indices, df_eval


def _print_metrics(df: pd.DataFrame, min_assay_size: int = MIN_ASSAY_SIZE) -> None:
    """Print size-weighted Pearson r per split and overall.

    Args:
        df: DataFrame with columns: assay_chembl_id, standard_type, split,
            pchembl_value_filled, pred_pchembl.
        min_assay_size: Minimum compounds per (assay, standard_type) group to
            include in the metric.
    """
    assay_ids = [
        f"{a}_{s}" for a, s in zip(df["assay_chembl_id"], df["standard_type"])
    ]
    labels = torch.tensor(df["pchembl_value_filled"].to_numpy(dtype=np.float32))
    preds = torch.tensor(df["pred_pchembl"].to_numpy(dtype=np.float32))

    print("\n=== Evaluation metrics ===")
    for split in sorted(df["split"].unique()):
        mask = (df["split"] == split).values
        split_assay_ids = [aid for aid, m in zip(assay_ids, mask) if m]
        r, _ = pearson_r_per_assay(
            preds[mask], labels[mask], split_assay_ids, min_assay_size=min_assay_size
        )
        n_rows = int(mask.sum())
        print(f"  {split:25s}  Pearson r = {float(r):.4f}  (n_rows = {n_rows:,})")

    if df["split"].nunique() > 1:
        r_all, _ = pearson_r_per_assay(preds, labels, assay_ids, min_assay_size=min_assay_size)
        print(f"  {'overall':25s}  Pearson r = {float(r_all):.4f}  (n_rows = {len(df):,})")


def evaluate_splits(
    checkpoint_path: Path,
    data_dir: Path,
    activities_path: Path,
    targets_path: Path,
    splits: list[str],
    output_csv: Path,
    batch_size: int = 512,
) -> None:
    """Run inference on specified splits and write per-row predictions to CSV.

    Fingerprints and molecular properties are computed inline from
    ``canonical_smiles`` in activities.parquet. Target uniprot_ids are resolved
    by joining activities with targets.parquet on target_chembl_id.

    Args:
        checkpoint_path: Path to the Lightning checkpoint (.ckpt).
        data_dir: Training preprocessing directory (target_index.json, meta.json,
            mol_properties.npz, and optionally oov_target_mapping.json).
        activities_path: Path to activities.parquet.
        targets_path: Path to targets.parquet (used to join target_chembl_id →
            uniprot_id).
        splits: Split labels to evaluate (e.g. ``["test", "2024_not_novel"]``).
        output_csv: Destination path for the predictions CSV.
        batch_size: Inference batch size.
    """
    print(f"Loading activities from {activities_path}")
    activities_df = pd.read_parquet(activities_path)
    targets_df = pd.read_parquet(targets_path)[["target_chembl_id", "uniprot_id"]]

    df = activities_df[activities_df["split"].isin(splits)].copy()
    missing = set(splits) - set(df["split"].unique())
    if missing:
        print(f"WARNING: no rows found for splits: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No rows found for any of the requested splits: {splits}")

    null_rows = int(df["pchembl_value_filled"].isna().sum())
    if null_rows:
        print(f"WARNING: dropping {null_rows} rows with null pchembl_value_filled")
        df = df[df["pchembl_value_filled"].notna()].copy()

    df = df.merge(targets_df, on="target_chembl_id", how="left")
    no_uniprot = int(df["uniprot_id"].isna().sum())
    if no_uniprot:
        print(f"WARNING: dropping {no_uniprot} rows with unmapped target_chembl_id")
        df = df[df["uniprot_id"].notna()].copy()

    no_smiles = int(df["canonical_smiles"].isna().sum())
    if no_smiles:
        print(f"WARNING: dropping {no_smiles} rows with null canonical_smiles")
        df = df[df["canonical_smiles"].notna()].copy()

    print(f"Preprocessing {len(df):,} rows across splits: {splits}")
    fp_matrix, props_matrix, fp_indices, target_indices, std_type_indices, df_eval = \
        preprocess_activities_for_eval(df, data_dir)

    n_smiles_dropped = len(df) - len(df_eval)
    print(f"  {len(df_eval):,} rows after SMILES filtering ({n_smiles_dropped} dropped)")

    dataset = PredictDataset(
        fps_matrix=torch.tensor(fp_matrix),
        props_matrix=torch.tensor(props_matrix),
        fp_indices=fp_indices,
        target_indices=target_indices,
        standard_type_indices=std_type_indices,
        names=df_eval["ligand_chembl_id"].tolist(),
        uniprot_ids=df_eval["uniprot_id"].tolist(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Loading model from {checkpoint_path}")
    model = AffinityModel.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model.eval()

    print("Running inference...")
    all_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            fps_b, props_b, target_idx_b, std_type_idx_b, _, _ = batch
            preds = model(fps_b, props_b, target_idx_b, std_type_idx_b).squeeze(1)
            all_preds.append(preds.cpu())

    pred_pchembl = torch.cat(all_preds).numpy()

    output_df = df_eval[
        ["assay_chembl_id", "ligand_chembl_id", "uniprot_id",
         "standard_type", "split", "pchembl_value_filled"]
    ].copy()
    output_df["pred_pchembl"] = pred_pchembl

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv} ({len(output_df):,} rows)")

    _print_metrics(output_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate binding affinity baseline on held-out splits."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to Lightning checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Training preprocessing directory"
    )
    parser.add_argument(
        "--activities", required=True, help="Path to activities.parquet"
    )
    parser.add_argument(
        "--targets", required=True, help="Path to targets.parquet"
    )
    parser.add_argument(
        "--splits", nargs="+", required=True, help="Split name(s) to evaluate"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Inference batch size (default: 512)"
    )
    args = parser.parse_args()

    evaluate_splits(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        activities_path=Path(args.activities),
        targets_path=Path(args.targets),
        splits=args.splits,
        output_csv=Path(args.output),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
