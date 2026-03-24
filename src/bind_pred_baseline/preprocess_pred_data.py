"""Inline preprocessing for model prediction (no intermediate npz files)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bind_pred_baseline.constants import MOL_PROP_FEATURES, STANDARD_TYPE_INDEX
from bind_pred_baseline.preprocess_utils import (
    FEATURE_NAMES as PROP_FEATURE_NAMES,
    compute_fingerprints,
    compute_mol_properties,
    normalise_mol_properties,
    resolve_target_ids,
)


def preprocess_data_for_prediction(
    input_csv: Path,
    data_dir: Path,
    standard_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Preprocess an input CSV for model prediction.

    Reads the input CSV (columns: ligand_name, uniprot_id, smiles), computes fingerprints
    and mol properties, normalises using training scalers, and returns arrays ready
    for PredictDataset construction.

    Args:
        input_csv: Path to CSV with columns: ligand_name, uniprot_id, smiles.
        data_dir: Directory from training preprocessing (contains
            target_index.json, meta.json, mol_properties.npz, and optionally
            oov_target_mapping.json).
        standard_type: Assay type string ("IC50", "Ki", or "Kd").

    Returns:
        Tuple of:
            fp_matrix:         float32 (N_unique_compounds, fp_size)
            props_matrix:      float32 (N_unique_compounds, n_mol_prop_features)
            fp_indices:        int64   (N_rows,)
            target_indices:    int64   (N_rows,)
            std_type_indices:  int64   (N_rows,) — all same value
            names:             list[str] (N_rows,)
            uniprot_ids:       list[str] (N_rows,)
    """
    # ------------------------------------------------------------------
    # Load input CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(input_csv)

    # ------------------------------------------------------------------
    # Load training target index, meta, and optional OOV mapping
    # ------------------------------------------------------------------
    with open(data_dir / "target_index.json") as f:
        target_index: dict[str, int] = json.load(f)

    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    fp_size: int = meta["fp_size"]
    fp_type: str = meta["fp_type"]

    oov_mapping_path = data_dir / "oov_target_mapping.json"
    if oov_mapping_path.exists():
        with open(oov_mapping_path) as f:
            oov_mapping: dict[str, str] = json.load(f)
    else:
        oov_mapping = {}

    # ------------------------------------------------------------------
    # Validate all uniprot_ids can be resolved
    # ------------------------------------------------------------------
    unresolvable = sorted(
        uid for uid in df["uniprot_id"].unique()
        if uid not in target_index and oov_mapping.get(uid, uid) not in target_index
    )
    if unresolvable:
        raise KeyError(
            f"Cannot resolve these uniprot_ids to any training target: {unresolvable}"
        )

    # ------------------------------------------------------------------
    # Compute fingerprints
    # ------------------------------------------------------------------
    unique_names = df["ligand_name"].tolist()
    unique_smiles = df["smiles"].tolist()

    fp_names, fp_matrix_raw = compute_fingerprints(
        mol_names=unique_names,
        smiles=unique_smiles,
        fp_type=fp_type,
        fp_size=fp_size,
    )

    # ------------------------------------------------------------------
    # Compute mol properties
    # ------------------------------------------------------------------
    prop_names, raw_props = compute_mol_properties(
        mol_names=unique_names,
        smiles=unique_smiles,
    )

    # ------------------------------------------------------------------
    # Jointly filter to compounds with both representations
    # ------------------------------------------------------------------
    prop_name_to_row: dict[str, int] = {name: i for i, name in enumerate(prop_names)}
    common = set(fp_names) & prop_name_to_row.keys()

    keep_mask = np.array([n in common for n in fp_names])
    fp_names_kept = fp_names[keep_mask]
    fp_matrix_kept = fp_matrix_raw[keep_mask].astype(np.float32)
    raw_props_kept = raw_props[[prop_name_to_row[n] for n in fp_names_kept]]

    # ------------------------------------------------------------------
    # Normalise mol properties using training scaler
    # ------------------------------------------------------------------
    train_props_npz = np.load(data_dir / "mol_properties.npz")
    normed_props = normalise_mol_properties(
        raw_props_kept,
        mean=train_props_npz["mean"],
        std=train_props_npz["std"],
    )

    # Select only MOL_PROP_FEATURES subset (same as train.py setup)
    feature_names = list(PROP_FEATURE_NAMES)
    col_indices = [feature_names.index(f) for f in MOL_PROP_FEATURES]
    props_matrix = normed_props[:, col_indices].astype(np.float32)

    # ------------------------------------------------------------------
    # Build index arrays for each row in the (possibly filtered) df
    # ------------------------------------------------------------------
    # Only keep rows whose compound name survived the joint filter
    df_filtered = df[df["ligand_name"].isin(common)].copy()

    fp_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(fp_names_kept)}

    fp_indices = np.array([fp_name_to_idx[n] for n in df_filtered["ligand_name"]], dtype=np.int64)
    target_indices = np.array(
        resolve_target_ids(df_filtered["uniprot_id"].tolist(), target_index, mapping=oov_mapping or None),
        dtype=np.int64,
    )
    std_type_value = STANDARD_TYPE_INDEX[standard_type]
    std_type_indices = np.full(len(df_filtered), std_type_value, dtype=np.int64)

    names: list[str] = df_filtered["ligand_name"].tolist()
    uniprot_ids: list[str] = df_filtered["uniprot_id"].tolist()

    return fp_matrix_kept, props_matrix, fp_indices, target_indices, std_type_indices, names, uniprot_ids
