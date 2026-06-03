"""Compound similarity computation for time-based benchmark splits.

This module covers all three stages of similarity assessment:
1. ECFP4 fingerprint computation (``compute_ecfp4_fingerprints``)
2. Tanimoto similarity filtering (``filter_by_tanimoto``)
3. Per-cutoff similarity labelling (``compute_similarity_for_cutoff_year``)
"""

import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from tqdm import tqdm


# ---------------------------------------------------------------------------
# ECFP4 fingerprint computation
# ---------------------------------------------------------------------------


def _process_one(args: tuple[str, str]) -> tuple[str, np.ndarray] | None:
    """Parse one SMILES and return (name, fingerprint) or None on failure.

    Must be a module-level function so it can be pickled for multiprocessing.
    """
    name, smi = args
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    if mol is None:
        print(f"WARNING: could not parse SMILES, skipping. name={name}, smiles={smi}")
        return None
    mol = AllChem.RemoveAllHs(mol)
    return name, fpgen.GetFingerprintAsNumPy(mol)


def compute_ecfp4_fingerprints(
    mol_names: list[str],
    smiles: list[str],
    n_jobs: int = 1,
    chunksize: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ECFP4 fingerprints (radius=2, size=2048) for a list of molecules.

    Molecules that fail to parse are skipped and excluded from the output.

    Preprocessing applied:
    - Sanitization via RDKit (aromaticity, valence checking)
    - Explicit Hs removed
    - Note: multi-component SMILES (e.g. salts like "CC(=O)[O-].[Na+]") are NOT
      stripped to the largest fragment. ~4% of ChEMBL SMILES are affected. This is
      a conscious choice to keep the preprocessing simple; be aware that counterion
      fragments will influence the fingerprint for those molecules.

    Args:
        mol_names: Molecule identifiers, one per SMILES.
        smiles: SMILES strings.
        n_jobs: Number of worker processes. 1 = single-process (default).
            -1 = use all available CPUs (``multiprocessing.cpu_count()``).
        chunksize: Number of molecules sent to each worker at a time.
            Larger values reduce IPC overhead but increase memory per worker.

    Returns:
        Tuple of (names, fingerprints) as numpy arrays, with failed parses removed.
        names shape: (N,), fingerprints shape: (N, 2048).
    """
    pairs = list(zip(mol_names, smiles))

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs == 1:
        results = [_process_one(p) for p in tqdm(pairs, desc="Computing ECFP4")]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_one, pairs, chunksize=chunksize),
                    total=len(pairs),
                    desc=f"Computing ECFP4 ({n_jobs} workers)",
                )
            )

    out_names = []
    out_fps = []
    for r in results:
        if r is not None:
            out_names.append(r[0])
            out_fps.append(r[1])

    return np.array(out_names), np.array(out_fps)


# ---------------------------------------------------------------------------
# Tanimoto similarity filter
# ---------------------------------------------------------------------------


def _np_to_bitvect(fp: np.ndarray) -> ExplicitBitVect:
    """Convert a 1D uint8 numpy array to an RDKit ExplicitBitVect."""
    ebv = ExplicitBitVect(len(fp))
    for idx in np.where(fp == 1)[0]:
        ebv.SetBit(int(idx))
    return ebv


def _compute_max_similarity(
    args: tuple[int, ExplicitBitVect],
    reference_bvs: list[ExplicitBitVect],
    reference_ids: np.ndarray,
) -> tuple[int, float, str | None]:
    """Return (candidate_index, max_similarity, most_similar_reference_id)."""
    idx, candidate_bv = args
    if len(reference_bvs) == 0:
        return idx, 0.0, None
    similarities = DataStructs.BulkTanimotoSimilarity(candidate_bv, reference_bvs)
    best = int(np.argmax(similarities))
    return idx, float(similarities[best]), reference_ids[best]


def filter_by_tanimoto(
    candidate_fps: np.ndarray,
    reference_fps: np.ndarray,
    reference_ids: np.ndarray,
    threshold: float,
    n_jobs: int = 1,
    chunksize: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each candidate fingerprint, compute the maximum Tanimoto similarity
    against all reference fingerprints.

    Args:
        candidate_fps: Binary fingerprints to evaluate, shape (N, fp_size).
        reference_fps: Binary fingerprints to compare against, shape (M, fp_size).
        reference_ids: Identifiers for each reference fingerprint, shape (M,).
        threshold: A candidate is considered novel if its max similarity is
            strictly less than this value.
        n_jobs: Number of worker processes. 1 = single-process (default).
            -1 = use all available CPUs.
        chunksize: Number of candidates sent to each worker at a time.

    Returns:
        Tuple of three arrays, all of length N:
        - is_novel: bool array, True where max similarity < threshold.
        - max_similarities: float array of max Tanimoto similarity per candidate.
        - most_similar_ids: array of the reference ID with highest similarity
            per candidate (None if reference set is empty).
    """
    if len(candidate_fps) == 0:
        return (
            np.empty(0, dtype=bool),
            np.empty(0, dtype=float),
            np.empty(0, dtype=object),
        )

    reference_bvs = [_np_to_bitvect(fp) for fp in reference_fps]
    candidate_bvs = [(i, _np_to_bitvect(fp)) for i, fp in enumerate(candidate_fps)]

    worker = partial(
        _compute_max_similarity,
        reference_bvs=reference_bvs,
        reference_ids=reference_ids,
    )

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs == 1:
        results = [worker(item) for item in tqdm(candidate_bvs, desc="Tanimoto filter")]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(worker, candidate_bvs, chunksize=chunksize),
                    total=len(candidate_bvs),
                    desc=f"Tanimoto filter ({n_jobs} workers)",
                )
            )

    # pool.imap preserves order, but sort defensively in case this is
    # ever switched to imap_unordered.
    results.sort(key=lambda r: r[0])

    max_similarities = np.array([r[1] for r in results], dtype=float)
    most_similar_ids = np.array([r[2] for r in results], dtype=object)
    is_novel = max_similarities < threshold

    return is_novel, max_similarities, most_similar_ids


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_similarity_for_cutoff_year(
    compounds_df: pd.DataFrame,
    cutoff_year: int,
    fp_index: dict[str, int],
    fp_matrix: np.ndarray,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute Tanimoto similarity for compounds relative to a time-based cutoff year.

    Compounds first seen before cutoff_year form the reference set and receive
    max_sim=1.0 and most_sim_cpd pointing to themselves (they are by definition
    identical to at least one pre-cutoff compound: themselves).  Compounds first
    seen >= cutoff_year are evaluated against the reference set via ECFP4 Tanimoto
    similarity.

    Whether a compound is considered "novel" (i.e. max_sim < threshold) is a
    downstream concern and is not computed here — callers should derive the
    is_novel column from max_sim and their chosen threshold.

    Args:
        compounds_df: DataFrame with at least columns: chembl_id, cpd_earliest_year.
        cutoff_year: Split year. Reference = cpd_earliest_year < cutoff_year;
            candidates = cpd_earliest_year >= cutoff_year.
        fp_index: Mapping from chembl_id to row index in fp_matrix.
        fp_matrix: Binary fingerprint matrix, shape (N, fp_size).
        n_jobs: Number of worker processes for the Tanimoto computation.

    Returns:
        DataFrame indexed by chembl_id (same rows as compounds_df) with two columns:
        - max_sim_{cutoff_year}: float, 1.0 for reference-set compounds.
        - most_sim_cpd_pre_{cutoff_year}: str, own chembl_id for reference-set compounds.
    """
    col_sim = f"max_sim_pre_{cutoff_year}"
    col_most_sim = f"most_sim_cpd_pre_{cutoff_year}"

    indexed = compounds_df.set_index("chembl_id")

    result = pd.DataFrame(
        {col_sim: np.nan, col_most_sim: pd.NA},
        index=indexed.index,
    )
    result.index.name = "chembl_id"

    ref_ids = np.array(
        [
            cid
            for cid in indexed.index
            if cid in fp_index and indexed.loc[cid, "cpd_earliest_year"] < cutoff_year
        ]
    )

    # Reference compounds are identical to themselves (a pre-cutoff compound) → sim=1.0
    result.loc[ref_ids, col_sim] = 1.0
    result.loc[ref_ids, col_most_sim] = ref_ids

    cand_ids = np.array(
        [
            cid
            for cid in indexed.index
            if cid in fp_index and indexed.loc[cid, "cpd_earliest_year"] >= cutoff_year
        ]
    )

    if len(cand_ids) == 0:
        return result

    ref_fps = (
        fp_matrix[[fp_index[cid] for cid in ref_ids]]
        if len(ref_ids) > 0
        else np.empty((0, fp_matrix.shape[1]))
    )
    cand_fps = fp_matrix[[fp_index[cid] for cid in cand_ids]]

    _, max_sims, most_similar_ids = filter_by_tanimoto(
        candidate_fps=cand_fps,
        reference_fps=ref_fps,
        reference_ids=ref_ids,
        threshold=0.0,  # threshold unused; we only need max_sims and most_similar_ids
        n_jobs=n_jobs,
    )

    result.loc[cand_ids, col_sim] = max_sims
    result.loc[cand_ids, col_most_sim] = most_similar_ids

    return result
