import multiprocessing
from functools import partial

import numpy as np
from rdkit import DataStructs
from tqdm import tqdm
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


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
