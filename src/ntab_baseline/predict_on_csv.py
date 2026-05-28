"""Predict binding affinity from a user-supplied CSV of compounds.

Reads a CSV with columns (ligand_name, uniprot_id, smiles), runs the model,
and writes a predictions CSV in the unified output format.

Usage
-----
    python -m ntab_baseline.predict_on_csv \\
        --checkpoint out_baseline/lightning_logs/version_0/checkpoints/best.ckpt \\
        --input-csv compounds.csv \\
        --output-csv predictions.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ntab_baseline.callbacks import resolve_data_dir
from ntab_baseline.constants import MIN_ASSAY_SIZE, STANDARD_TYPE_INDEX
from ntab_baseline.predict_on_benchmark import (
    _assemble_output_df,
    _print_metrics,
    run_inference,
)
from ntab_baseline.model import AffinityModel
from ntab_baseline.preprocess_pred_data import preprocess_for_inference


def load_csv_as_standard_df(
    input_csv: Path,
    default_standard_type: str = "IC50",
) -> pd.DataFrame:
    """Read a user CSV and return a standard input DataFrame.

    Accepts columns: ligand_name, uniprot_id, smiles (or canonical_smiles),
    and optionally standard_type and pchembl_value. If standard_type is absent,
    fills it from default_standard_type. Sets split = "predict". pchembl_value
    is passed through if present.

    Args:
        input_csv: Path to input CSV.
        default_standard_type: Assay type applied to all rows when the CSV
            does not contain a standard_type column (default: "IC50").

    Returns:
        Standard input DataFrame with columns: ligand_name, smiles, uniprot_id,
        standard_type, split, and optionally pchembl_value.
    """
    df = pd.read_csv(input_csv)

    # Normalise SMILES column name
    if "canonical_smiles" in df.columns and "smiles" not in df.columns:
        df = df.rename(columns={"canonical_smiles": "smiles"})

    # Fill standard_type if not provided per-row, then normalise case
    # (e.g. "ic50" → "IC50", "ki" → "Ki", "kd" → "Kd")
    _ST_NORM = {k.lower(): k for k in STANDARD_TYPE_INDEX}
    if "standard_type" not in df.columns:
        df = df.copy()
        df["standard_type"] = default_standard_type
    else:
        df["standard_type"] = (
            df["standard_type"].str.lower().map(_ST_NORM).fillna(df["standard_type"])
        )

    df["split"] = "predict"

    return df


def predict_on_csv(
    checkpoint_path: Path,
    input_csv: Path,
    output_csv: Path,
    standard_type: str = "IC50",
    batch_size: int = 256,
    n_bootstrap: int | None = None,
    device: str | None = None,
    weighted: bool = False,
    min_assay_size: int = MIN_ASSAY_SIZE,
    extra_oov_mapping_file: Path | None = None,
) -> None:
    """Run model inference on a user CSV and write predictions.

    Args:
        checkpoint_path: Path to the Lightning checkpoint (.ckpt).
        input_csv: Path to input CSV (columns: ligand_name, uniprot_id, smiles).
        output_csv: Destination path for the predictions CSV.
        standard_type: Assay type applied to all rows when the CSV does not
            contain a standard_type column (default: "IC50").
        batch_size: Inference batch size.
        n_bootstrap: If given and labels are present, compute bootstrapped SE.
        device: Torch device. Defaults to "cuda" if available, else "cpu".
        weighted: If False (default), macro-average Pearson r. If True, size-weighted.
        min_assay_size: Minimum compounds per assay to include in metrics.
        extra_oov_mapping_file: Optional path to a JSON file with extra OOV target
            mappings, merged on top of the checkpoint's oov_target_mapping.json.
    """
    data_dir = resolve_data_dir(checkpoint_path)

    extra_oov_mapping: dict[str, str] | None = None
    if extra_oov_mapping_file is not None:
        with open(extra_oov_mapping_file) as f:
            extra_oov_mapping = json.load(f)

    print(f"Loading input from {input_csv}")
    df = load_csv_as_standard_df(input_csv, default_standard_type=standard_type)

    print(f"Preprocessing {len(df):,} rows...")
    (
        fp_matrix,
        props_matrix,
        fp_indices,
        target_indices,
        std_type_indices,
        df_filtered,
    ) = preprocess_for_inference(df, data_dir, extra_oov_mapping=extra_oov_mapping)

    n_dropped = len(df) - len(df_filtered)
    print(f"  {len(df_filtered):,} rows after SMILES filtering ({n_dropped} dropped)")

    print(f"Loading model from {checkpoint_path}")
    model = AffinityModel.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    smiles_list: list[str] | None = None
    if getattr(model.hparams, "use_chemprop", False):
        smiles_list = df_filtered["smiles"].tolist()

    print("Running inference...")
    pred_pchembl = run_inference(
        model,
        fp_matrix,
        props_matrix,
        fp_indices,
        target_indices,
        std_type_indices,
        batch_size=batch_size,
        device=device,
        smiles=smiles_list,
    )

    output_df = _assemble_output_df(df_filtered, pred_pchembl)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv} ({len(output_df):,} rows)")

    if output_df["pchembl_value"].notna().any():
        _print_metrics(output_df, min_assay_size=min_assay_size, n_bootstrap=n_bootstrap, weighted=weighted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict binding affinity for compounds in a CSV file."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to Lightning checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV with columns: ligand_name, uniprot_id, smiles",
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--standard-type",
        default="IC50",
        choices=["IC50", "Ki", "Kd"],
        help="Assay type applied to all rows when not in CSV (default: IC50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size (default: 256)",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=None,
        help="Number of bootstrap resamples for SE on Pearson r (default: off)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: auto-detect GPU, fall back to CPU)",
    )
    parser.add_argument(
        "--size-weighted",
        action="store_true",
        default=False,
        help="Use size-weighted Pearson r instead of macro-average (default: macro)",
    )
    parser.add_argument(
        "--min-assay-size",
        type=int,
        default=MIN_ASSAY_SIZE,
        help=f"Minimum compounds per assay for metrics (default: {MIN_ASSAY_SIZE})",
    )
    parser.add_argument(
        "--extra-oov-mapping-file",
        default=None,
        help="Path to a JSON file mapping unknown UniProt IDs (not seen during training) to "
             "the most similar training target. Format: {\"Q99999\": \"P00000\", ...}. "
             "Merged on top of any mappings saved with the checkpoint.",
    )
    args = parser.parse_args()

    predict_on_csv(
        checkpoint_path=Path(args.checkpoint),
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        standard_type=args.standard_type,
        batch_size=args.batch_size,
        n_bootstrap=args.n_bootstraps,
        device=args.device,
        weighted=args.size_weighted,
        min_assay_size=args.min_assay_size,
        extra_oov_mapping_file=Path(args.extra_oov_mapping_file) if args.extra_oov_mapping_file else None,
    )


if __name__ == "__main__":
    main()
