"""Data loading, filtering, preprocessing utilities, fingerprint and property computation."""

import multiprocessing
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

# Note: multi-component SMILES (e.g. salts like "CC(=O)[O-].[Na+]") are NOT
# stripped to the largest fragment. ~4% of ChEMBL SMILES are affected — counterion
# fragments will influence the fingerprint for those molecules.


def _process_one(
    args: tuple[str, str, str, int, int],
) -> tuple[str, np.ndarray] | None:
    """Parse one SMILES and return (name, fingerprint) or None on failure.

    Must be a module-level function so it can be pickled for multiprocessing.

    Args:
        args: Tuple of (name, smiles, fp_type, radius, fp_size).
    """
    name, smi, fp_type, radius, fp_size = args
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    if smi is None:
        print(f"WARNING: null SMILES, skipping. name={name}")
        return None
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    if mol is None:
        print(f"WARNING: could not parse SMILES, skipping. name={name}, smiles={smi}")
        return None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = AllChem.RemoveAllHs(mol)
    if fp_type == "count":
        return name, fpgen.GetCountFingerprintAsNumPy(mol)
    return name, fpgen.GetFingerprintAsNumPy(mol)


def compute_fingerprints(
    mol_names: list[str],
    smiles: list[str],
    fp_type: Literal["binary", "count"] = "binary",
    radius: int = 2,
    fp_size: int = 2048,
    n_jobs: int = 1,
    chunksize: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Morgan fingerprints for a list of molecules.

    Molecules that fail to parse are skipped and excluded from the output.

    Args:
        mol_names: Molecule identifiers, one per SMILES.
        smiles: SMILES strings.
        fp_type: "binary" (bit vector, 0/1) or "count" (occurrence counts per substructure).
        radius: Morgan radius. radius=2 corresponds to ECFP4.
        fp_size: Number of bits in the fingerprint vector.
        n_jobs: Number of worker processes. 1 = single-process. -1 = all CPUs.
        chunksize: Molecules per worker chunk (affects multiprocessing IPC overhead).

    Returns:
        Tuple of (names, fingerprints) as numpy arrays, with failed parses removed.
        names shape: (N,), fingerprints shape: (N, fp_size).
    """
    pairs = [
        (name, smi, fp_type, radius, fp_size) for name, smi in zip(mol_names, smiles)
    ]

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    desc = f"Computing ECFP{radius * 2} ({fp_type})"
    if n_jobs == 1:
        results = [_process_one(p) for p in tqdm(pairs, desc=desc)]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_one, pairs, chunksize=chunksize),
                    total=len(pairs),
                    desc=f"{desc}, {n_jobs} workers",
                )
            )

    out_names: list[str] = []
    out_fps: list[np.ndarray] = []
    for r in results:
        if r is not None:
            out_names.append(r[0])
            out_fps.append(r[1])

    return np.array(out_names), np.array(out_fps)


# ---------------------------------------------------------------------------
# Mol properties
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "MolLogP",
    "ExactMolWt",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "FormalCharge",
    "MolMR",
    "FractionCSP3",
    "RingCount",
    "NumAromaticRings",
    "HeavyAtomCount",
]


def _compute_one(args: tuple[str, str]) -> tuple[str, np.ndarray] | None:
    """Parse one SMILES and return (name, properties) or None on failure.

    Module-level so it can be pickled for multiprocessing.
    """
    name, smi = args
    if smi is None:
        print(f"WARNING: null SMILES, skipping. name={name}")
        return None
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    if mol is None:
        print(f"WARNING: could not parse SMILES, skipping. name={name}, smiles={smi}")
        return None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = Chem.RemoveAllHs(mol)

    props = np.array(
        [
            Descriptors.MolLogP(mol),
            Descriptors.ExactMolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            float(Chem.rdmolops.GetFormalCharge(mol)),
            Descriptors.MolMR(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.RingCount(mol),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(mol.GetNumHeavyAtoms()),
        ],
        dtype=np.float32,
    )
    return name, props


def compute_mol_properties(
    mol_names: list[str],
    smiles: list[str],
    n_jobs: int = 1,
    chunksize: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute physicochemical properties for a list of molecules.

    Molecules that fail to parse are skipped and excluded from the output.
    Salt spectators are stripped to the largest organic fragment before
    descriptor computation (same behaviour as compute_fingerprints).

    Args:
        mol_names: Molecule identifiers, one per SMILES.
        smiles: SMILES strings.
        n_jobs: Number of worker processes. 1 = single-process. -1 = all CPUs.
        chunksize: Molecules per worker chunk.

    Returns:
        Tuple of (names, props_matrix) as numpy arrays, with failed parses removed.
        names shape: (N,), props_matrix shape: (N, 12), dtype float32.
    """
    pairs = [(name, smi) for name, smi in zip(mol_names, smiles)]

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs == 1:
        results = [
            _compute_one(p) for p in tqdm(pairs, desc="Computing mol properties")
        ]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(_compute_one, pairs, chunksize=chunksize),
                    total=len(pairs),
                    desc=f"Computing mol properties, {n_jobs} workers",
                )
            )

    out_names: list[str] = []
    out_props: list[np.ndarray] = []
    for r in results:
        if r is not None:
            out_names.append(r[0])
            out_props.append(r[1])

    if not out_props:
        return np.array(out_names), np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    return np.array(out_names), np.stack(out_props)


def normalise_mol_properties(
    props: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalise a property matrix using pre-computed mean and std.

    Constant features (std == 0) are mapped to 0.0 rather than NaN.

    Args:
        props: Raw property matrix, shape (N, 12), float32.
        mean: Per-feature mean to subtract, shape (12,).
        std: Per-feature std to divide by, shape (12,).

    Returns:
        Normalised property matrix, shape (N, 12), float32.
    """
    props = props.astype(np.float32)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    std_safe = np.where(std == 0.0, 1.0, std).astype(np.float32)
    return ((props - mean) / std_safe).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading, filtering, and preprocessing
# ---------------------------------------------------------------------------


def load_activities(path: str | Path) -> pd.DataFrame:
    """Load activities.parquet and filter to IC50 measurements with relation '=' on wild-type targets.

    Drops rows where standard_type != IC50, pchembl_relation != '=', or mutation is not null
    (i.e. mutant-target assays are excluded).

    Args:
        path: Path to activities.parquet.

    Returns:
        Filtered DataFrame.
    """
    df = pd.read_parquet(path)
    df = df[
        df["standard_type"].isin(["IC50", "Ki", "Kd"]) & (df["pchembl_relation"] == "=")
    ]
    if "mutation" in df.columns:
        df = df[df["mutation"].isna()]
    df = df[df["pchembl_value_filled"].notna()]
    return df.reset_index(drop=True)


def average_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Average pchembl_value_filled per (target_chembl_id, ligand_chembl_id) for train rows.

    Only train-split rows are collapsed; all other splits are returned unchanged.

    Args:
        df: Activities DataFrame with at least columns: target_chembl_id,
            ligand_chembl_id, pchembl_value_filled, split.

    Returns:
        DataFrame with train duplicates averaged; non-train rows untouched.
    """
    train_mask = df["split"] == "train"
    train = df[train_mask]
    other = df[~train_mask]

    group_keys = ["target_chembl_id", "ligand_chembl_id", "standard_type"]
    pchembl_mean = train.groupby(group_keys, as_index=False)[
        "pchembl_value_filled"
    ].mean()
    # Preserve all other columns by taking the first occurrence per group,
    # then overwrite pchembl_value_filled with the computed mean.
    train_first = train.groupby(group_keys, as_index=False).first()
    train_averaged = train_first.drop(columns=["pchembl_value_filled"]).merge(
        pchembl_mean, on=group_keys
    )

    return pd.concat([train_averaged, other], ignore_index=True)


def build_target_index(target_ids: list[str] | pd.Series) -> dict[str, int]:
    """Build a deterministic {target_chembl_id: int} index from training target IDs.

    Args:
        target_ids: Target IDs seen during training (duplicates allowed).

    Returns:
        Dict mapping each unique target ID to a contiguous integer index starting at 0.
    """
    unique = sorted(set(target_ids))
    return {tid: i for i, tid in enumerate(unique)}


def _blosum62_best_match(
    args: tuple[str, str | None, list[tuple[str, str]], str],
) -> tuple[str, str]:
    """Worker function for parallel OOV alignment.

    Creates its own PairwiseAligner so it can be safely used across processes.

    Args:
        args: (oov_id, sanitized_oov_seq, train_with_seq, fallback)

    Returns:
        (oov_id, best_matching_train_id)
    """
    oov_id, oov_seq, train_with_seq, fallback = args
    if not oov_seq or not train_with_seq:
        return oov_id, fallback
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"
    best_id, best_score = fallback, float("-inf")
    for tid, tseq in train_with_seq:
        score = aligner.score(oov_seq, tseq)
        if score > best_score:
            best_score = score
            best_id = tid
    return oov_id, best_id


def find_closest_training_targets(
    oov_ids: list[str],
    train_ids: list[str],
    sequences: dict[str, str | None],
    n_jobs: int = 1,
    train_counts: dict[str, int] | None = None,
    min_train_datapoints: int = 0,
) -> dict[str, str]:
    """For each OOV target, find the most sequence-similar training target.

    Uses BLOSUM62 global pairwise alignment (Biopython PairwiseAligner).
    Training targets with no sequence are excluded from consideration.
    OOV targets with no sequence are assigned the first available training target.
    Non-standard amino acid characters (e.g. selenocysteine 'U') are replaced
    with glycine ('G') before alignment; a warning is printed for each affected target.

    Args:
        oov_ids: Uniprot IDs of targets not in the training index.
        train_ids: Uniprot IDs of all training targets.
        sequences: Dict mapping uniprot_id → protein sequence (or None).
        n_jobs: Number of parallel worker processes. -1 uses all available CPUs.
        train_counts: Dict mapping uniprot_id → number of training datapoints.
            Used together with min_train_datapoints to filter eligible targets.
        min_train_datapoints: Minimum number of training datapoints a target must
            have to be eligible as an OOV mapping destination. Ignored if 0 or
            train_counts is not provided.

    Returns:
        Dict mapping each oov_id to the best matching train_id.
    """
    if not oov_ids:
        return {}

    matrix = substitution_matrices.load("BLOSUM62")
    valid_chars = set(matrix.alphabet)

    def _sanitize(seq: str) -> str:
        return "".join(c if c in valid_chars else "G" for c in seq)

    def _has_nonstandard(seq: str) -> bool:
        return any(c not in valid_chars for c in seq)

    eligible_train_ids = train_ids
    if train_counts and min_train_datapoints > 0:
        eligible_train_ids = [
            tid for tid in train_ids if train_counts.get(tid, 0) >= min_train_datapoints
        ]
        if not eligible_train_ids:
            print(
                f"  Warning: no training targets meet min_train_datapoints={min_train_datapoints}, falling back to all targets"
            )
            eligible_train_ids = train_ids
        else:
            print(
                f"  {len(eligible_train_ids)}/{len(train_ids)} training targets eligible (>= {min_train_datapoints} datapoints)"
            )

    train_with_seq = [
        (tid, _sanitize(sequences[tid]))
        for tid in eligible_train_ids
        if sequences.get(tid)
    ]
    fallback = eligible_train_ids[0]

    # Sanitize OOV sequences upfront (single-threaded) so warnings are printed cleanly
    sanitized_oov: dict[str, str | None] = {}
    for oov_id in oov_ids:
        seq = sequences.get(oov_id)
        if seq and _has_nonstandard(seq):
            print(
                f"  Warning: {oov_id} has non-standard amino acid characters — sanitizing with glycine ('G') before alignment"
            )
            seq = _sanitize(seq)
        sanitized_oov[oov_id] = seq

    worker_args = [
        (oov_id, sanitized_oov[oov_id], train_with_seq, fallback) for oov_id in oov_ids
    ]

    effective_jobs = os.cpu_count() if n_jobs == -1 else n_jobs

    result: dict[str, str] = {}
    if effective_jobs == 1:
        for args in tqdm(worker_args, desc="Aligning OOV targets"):
            oov_id, best_id = _blosum62_best_match(args)
            result[oov_id] = best_id
    else:
        with multiprocessing.Pool(effective_jobs) as pool:
            for oov_id, best_id in tqdm(
                pool.imap(_blosum62_best_match, worker_args),
                total=len(worker_args),
                desc="Aligning OOV targets",
            ):
                result[oov_id] = best_id
    return result


def load_split_from_file(path: str | Path) -> dict[str, str]:
    """Load a uniprot_id → split mapping from a CSV file.

    The CSV must have columns 'uniprot_id' and 'data_split'.

    Args:
        path: Path to the CSV file.

    Returns:
        Dict mapping uniprot_id to split string (e.g. 'train', 'val', 'test').
    """
    df = pd.read_csv(path)
    if "uniprot_id" not in df.columns or "data_split" not in df.columns:
        raise ValueError(
            f"split_from_file CSV must have 'uniprot_id' and 'data_split' columns, got: {list(df.columns)}"
        )
    return dict(zip(df["uniprot_id"], df["data_split"]))


def resolve_target_ids(
    target_ids: list[str],
    index: dict[str, int],
    mapping: dict[str, str] | None = None,
) -> list[int]:
    """Map target IDs to integer indices, with optional remapping for OOV targets.

    Args:
        target_ids: Target IDs to resolve (e.g. from val/test split).
        index: Index built from training data via build_target_index.
        mapping: Optional dict of {oov_target_id: train_target_id} to remap
            targets not present in the training index. Useful for manually
            substituting the closest known target for an unseen one.

    Returns:
        List of integer indices, one per input target ID.

    Raises:
        KeyError: If a target ID (after applying mapping) is not in the index.
    """
    resolved = []
    for tid in target_ids:
        mapped = mapping.get(tid, tid) if mapping else tid
        if mapped not in index:
            raise KeyError(
                f"Target '{tid}' (mapped to '{mapped}') not found in training index. "
                "Add it to target_mapping in data.yaml to remap it to a known target."
            )
        resolved.append(index[mapped])
    return resolved
