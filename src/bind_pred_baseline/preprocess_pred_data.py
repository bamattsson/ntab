"""Shared preprocessing for model inference.

Provides:
- preprocess_for_inference: DataFrame-in, arrays-out. Accepts the standard
  input DataFrame format (ligand_name, smiles, uniprot_id, standard_type).
"""

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


def preprocess_for_inference(
    df: pd.DataFrame,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Preprocess a standard input DataFrame for model inference.

    Computes fingerprints and molecular properties from SMILES, normalises
    using training scalers, and resolves target and standard-type indices from
    training artifacts. Rows where SMILES fails to parse are dropped with a
    warning.

    Args:
        df: Standard input DataFrame with required columns:
            - ligand_name: compound identifier used for deduplication
            - smiles: SMILES string (also accepts canonical_smiles)
            - uniprot_id: target UniProt accession
            - standard_type: per-row assay type ("IC50", "Ki", or "Kd")
            Any additional columns are preserved in df_filtered.
        data_dir: Training preprocessing directory containing target_index.json,
            meta.json, mol_properties.npz, and optionally oov_target_mapping.json.

    Returns:
        Tuple of:
            fp_matrix:        float32 array (N_unique_compounds, fp_size)
            props_matrix:     float32 array (N_unique_compounds, n_mol_prop_features)
            fp_indices:       int64 array   (N_rows,)
            target_indices:   int64 array   (N_rows,)
            std_type_indices: int64 array   (N_rows,) — per-row standard type
            df_filtered:      DataFrame     (N_rows,) same row order as the
                              arrays above, with the same columns as df.
    """
    # Support canonical_smiles as an alias for smiles
    if "canonical_smiles" in df.columns and "smiles" not in df.columns:
        df = df.rename(columns={"canonical_smiles": "smiles"})

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

    # Deduplicate by ligand_name for fingerprint/property computation
    unique_cpds = df[["ligand_name", "smiles"]].drop_duplicates("ligand_name")

    fp_names, fp_matrix_raw = compute_fingerprints(
        mol_names=unique_cpds["ligand_name"].tolist(),
        smiles=unique_cpds["smiles"].tolist(),
        fp_type=fp_type,
        fp_size=fp_size,
    )

    prop_names, raw_props = compute_mol_properties(
        mol_names=unique_cpds["ligand_name"].tolist(),
        smiles=unique_cpds["smiles"].tolist(),
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

    df_filtered = df[df["ligand_name"].isin(common)].reset_index(drop=True)

    fp_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(fp_names_kept)}

    fp_indices = np.array(
        [fp_name_to_idx[n] for n in df_filtered["ligand_name"]], dtype=np.int64
    )
    target_indices = np.array(
        resolve_target_ids(
            df_filtered["uniprot_id"].tolist(), target_index, mapping=oov_mapping or None
        ),
        dtype=np.int64,
    )
    std_type_indices = np.array(
        [STANDARD_TYPE_INDEX[st] for st in df_filtered["standard_type"]], dtype=np.int64
    )

    return fp_matrix_kept, props_matrix, fp_indices, target_indices, std_type_indices, df_filtered


