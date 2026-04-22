"""Evaluation script: run inference on held-out splits from activities.parquet.

Computes fingerprints and molecular properties inline from SMILES, respects
per-row standard_type, and writes a per-row predictions CSV alongside the
measured labels. Also prints size-weighted Pearson r per split.

Usage
-----
    python -m nfab_baseline.predict_on_benchmark \\
        --checkpoint out_baseline/lightning_logs/version_0/checkpoints/best.ckpt \\
        --data-dir out_baseline/data_preprocessing \\
        --activities out/activities.parquet \\
        --targets out/targets.parquet \\
        --splits test discard_not_novel \\
        --output predictions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from nfab_baseline.constants import MIN_ASSAY_SIZE
from nfab_baseline.model import AffinityModel
from nfab_baseline.model_utils import pearson_r_per_assay
from nfab_baseline.preprocess_pred_data import preprocess_for_inference

OUTPUT_COLUMNS = [
    "assay_id",
    "ligand_name",
    "uniprot_id",
    "standard_type",
    "split",
    "pchembl_value",
    "pred_pchembl",
]


def load_activities_as_standard_df(
    activities_path: Path,
    targets_path: Path,
    splits: list[str],
) -> pd.DataFrame:
    """Load activities and targets parquets and return a standard input DataFrame.

    Filters to the requested splits, joins uniprot_id from targets, drops rows
    with null pchembl_value_filled / uniprot_id / smiles, and renames columns
    to the standard format (assay_id, ligand_name, smiles, pchembl_value).

    Args:
        activities_path: Path to activities.parquet.
        targets_path: Path to targets.parquet (used to join target_chembl_id
            to uniprot_id).
        splits: Split names to include (e.g. ["test", "discard_not_novel"]).

    Returns:
        Standard input DataFrame with columns: assay_id, ligand_name, smiles,
        uniprot_id, standard_type, split, pchembl_value.
    """
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

    df = df.rename(
        columns={
            "assay_chembl_id": "assay_id",
            "ligand_chembl_id": "ligand_name",
            "canonical_smiles": "smiles",
            "pchembl_value_filled": "pchembl_value",
        }
    )

    return df


def run_inference(
    model: AffinityModel,
    fp_matrix: np.ndarray,
    props_matrix: np.ndarray,
    fp_indices: np.ndarray,
    target_indices: np.ndarray,
    std_type_indices: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """Run model inference and return predictions as a numpy array.

    Args:
        model: Trained AffinityModel.
        fp_matrix: float32 array (N_unique_compounds, fp_size).
        props_matrix: float32 array (N_unique_compounds, n_mol_prop_features).
        fp_indices: int64 array (N_rows,) — index into fp_matrix per row.
        target_indices: int64 array (N_rows,).
        std_type_indices: int64 array (N_rows,).
        batch_size: DataLoader batch size.
        device: Torch device string. Defaults to "cuda" if available, else "cpu".

    Returns:
        float32 numpy array of shape (N_rows,).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fp_t = torch.tensor(fp_matrix)
    props_t = torch.tensor(props_matrix)
    fp_idx_t = torch.tensor(fp_indices, dtype=torch.long)
    target_idx_t = torch.tensor(target_indices, dtype=torch.long)
    std_t = torch.tensor(std_type_indices, dtype=torch.long)

    # Gather per-row features from shared compound matrices
    fps_per_row = fp_t[fp_idx_t]
    props_per_row = props_t[fp_idx_t]

    dataset = TensorDataset(fps_per_row, props_per_row, target_idx_t, std_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    model.eval()

    all_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for fps_b, props_b, target_idx_b, std_type_idx_b in loader:
            fps_b = fps_b.to(device)
            props_b = props_b.to(device)
            target_idx_b = target_idx_b.to(device)
            std_type_idx_b = std_type_idx_b.to(device)
            preds = model(fps_b, props_b, target_idx_b, std_type_idx_b).squeeze(1)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds).numpy()


def _assemble_output_df(
    df_filtered: pd.DataFrame, pred_pchembl: np.ndarray
) -> pd.DataFrame:
    """Assemble the unified output DataFrame.

    Adds pred_pchembl, ensures split defaults to "predict" if absent, and
    ensures pchembl_value column is present (NaN if not in input).
    Returns only OUTPUT_COLUMNS in order.
    """
    df_out = df_filtered.copy()
    if "split" not in df_out.columns:
        df_out["split"] = "predict"
    if "pchembl_value" not in df_out.columns:
        df_out["pchembl_value"] = np.nan
    if "assay_id" not in df_out.columns:
        df_out["assay_id"] = df_out["uniprot_id"]
    df_out["pred_pchembl"] = pred_pchembl
    return df_out[OUTPUT_COLUMNS]


def _count_qualifying_assays(assay_ids: list[str], min_assay_size: int) -> int:
    """Count assay groups with at least min_assay_size samples."""
    counts: dict[str, int] = {}
    for aid in assay_ids:
        counts[aid] = counts.get(aid, 0) + 1
    return sum(1 for n in counts.values() if n >= min_assay_size)


def _print_metrics(
    df: pd.DataFrame,
    min_assay_size: int = MIN_ASSAY_SIZE,
    n_bootstrap: int | None = None,
    weighted: bool = False,
) -> None:
    """Print macro-averaged Pearson r per split and overall.

    Args:
        df: DataFrame with columns: assay_id, standard_type, split,
            pchembl_value, pred_pchembl.
        min_assay_size: Minimum compounds per (assay, standard_type) group to
            include in the metric.
        n_bootstrap: If given, also compute and print bootstrapped SE.
        weighted: If False (default), macro-average (equal weight per assay).
            If True, size-weighted average.
    """
    assay_ids = [f"{a}_{s}" for a, s in zip(df["assay_id"], df["standard_type"])]
    labels = df["pchembl_value"].to_numpy(dtype=np.float32)
    preds = df["pred_pchembl"].to_numpy(dtype=np.float32)

    print("\n=== Evaluation metrics ===")
    for split in sorted(df["split"].unique()):
        mask = (df["split"] == split).values
        split_assay_ids = [aid for aid, m in zip(assay_ids, mask) if m]
        r, ci_low, ci_high = pearson_r_per_assay(
            preds[mask],
            labels[mask],
            split_assay_ids,
            min_assay_size=min_assay_size,
            n_bootstrap=n_bootstrap,
            weighted=weighted,
        )
        n_rows = int(mask.sum())
        n_assays = _count_qualifying_assays(split_assay_ids, min_assay_size)
        ci_str = f" [{ci_low:.4f}, {ci_high:.4f}]" if ci_low is not None else ""
        print(
            f"  {split:25s}  Pearson r = {float(r):.4f}{ci_str}"
            f"  (n_rows = {n_rows:,}, n_assays = {n_assays})"
        )

    if df["split"].nunique() > 1:
        n_assays_all = _count_qualifying_assays(assay_ids, min_assay_size)
        r_all, ci_low_all, ci_high_all = pearson_r_per_assay(
            preds,
            labels,
            assay_ids,
            min_assay_size=min_assay_size,
            n_bootstrap=n_bootstrap,
            weighted=weighted,
        )
        ci_str = f" [{ci_low_all:.4f}, {ci_high_all:.4f}]" if ci_low_all is not None else ""
        print(
            f"  {'overall':25s}  Pearson r = {float(r_all):.4f}{ci_str}"
            f"  (n_rows = {len(df):,}, n_assays = {n_assays_all})"
        )


def evaluate_splits(
    checkpoint_path: Path,
    data_dir: Path,
    activities_path: Path,
    targets_path: Path,
    splits: list[str],
    output_csv: Path,
    batch_size: int = 512,
    n_bootstrap: int | None = None,
    weighted: bool = False,
) -> None:
    """Run inference on specified splits and write per-row predictions to CSV.

    Fingerprints and molecular properties are computed inline from SMILES in
    activities.parquet. Target uniprot_ids are resolved by joining activities
    with targets.parquet on target_chembl_id.

    Args:
        checkpoint_path: Path to the Lightning checkpoint (.ckpt).
        data_dir: Training preprocessing directory (target_index.json, meta.json,
            mol_properties.npz, and optionally oov_target_mapping.json).
        activities_path: Path to activities.parquet.
        targets_path: Path to targets.parquet (used to join target_chembl_id to
            uniprot_id).
        splits: Split labels to evaluate (e.g. ["test", "discard_not_novel"]).
        output_csv: Destination path for the predictions CSV.
        batch_size: Inference batch size.
        n_bootstrap: If given, compute bootstrapped SE for each Pearson r metric.
        weighted: If False (default), macro-average Pearson r. If True, size-weighted.
    """
    print(f"Loading activities from {activities_path}")
    df = load_activities_as_standard_df(activities_path, targets_path, splits)

    print(f"Preprocessing {len(df):,} rows across splits: {splits}")
    (
        fp_matrix,
        props_matrix,
        fp_indices,
        target_indices,
        std_type_indices,
        df_filtered,
    ) = preprocess_for_inference(df, data_dir)

    n_smiles_dropped = len(df) - len(df_filtered)
    print(
        f"  {len(df_filtered):,} rows after SMILES filtering ({n_smiles_dropped} dropped)"
    )

    print(f"Loading model from {checkpoint_path}")
    model = AffinityModel.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    print("Running inference...")
    pred_pchembl = run_inference(
        model,
        fp_matrix,
        props_matrix,
        fp_indices,
        target_indices,
        std_type_indices,
        batch_size=batch_size,
    )

    output_df = _assemble_output_df(df_filtered, pred_pchembl)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv} ({len(output_df):,} rows)")

    if output_df["pchembl_value"].notna().any():
        _print_metrics(output_df, n_bootstrap=n_bootstrap, weighted=weighted)


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
    parser.add_argument("--targets", required=True, help="Path to targets.parquet")
    parser.add_argument(
        "--splits", nargs="+", required=True, help="Split name(s) to evaluate"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size (default: 512)",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=None,
        help="Number of bootstrap resamples for SE on Pearson r (default: off)",
    )
    parser.add_argument(
        "--size-weighted",
        action="store_true",
        default=False,
        help="Use size-weighted Pearson r instead of macro-average (default: macro)",
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
        n_bootstrap=args.n_bootstraps,
        weighted=args.size_weighted,
    )


if __name__ == "__main__":
    main()
