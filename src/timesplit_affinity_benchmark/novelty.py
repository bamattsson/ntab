import numpy as np
import pandas as pd

from timesplit_affinity_benchmark.tanimoto_filter import filter_by_tanimoto


def compute_novelty_for_cutoff(
    compounds_df: pd.DataFrame,
    cutoff_year: int,
    fp_index: dict[str, int],
    fp_matrix: np.ndarray,
    threshold: float | None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute Tanimoto novelty for compounds relative to a time-based cutoff year.

    Compounds first seen before cutoff_year form the reference set and receive NaN
    for all output columns.  Compounds first seen >= cutoff_year are evaluated
    against the reference set via ECFP4 Tanimoto similarity.

    Args:
        compounds_df: DataFrame with at least columns: chembl_id, cpd_earliest_year.
        cutoff_year: Split year. Reference = cpd_earliest_year < cutoff_year;
            candidates = cpd_earliest_year >= cutoff_year.
        fp_index: Mapping from chembl_id to row index in fp_matrix.
        fp_matrix: Binary fingerprint matrix, shape (N, fp_size).
        threshold: A compound is novel if its max Tanimoto similarity is strictly
            less than this value.  Pass None to disable filtering — all candidates
            are marked as novel without running the Tanimoto computation.
        n_jobs: Number of worker processes for the Tanimoto computation.

    Returns:
        DataFrame indexed by chembl_id (same rows as compounds_df) with three columns:
        - is_novel_{cutoff_year}: bool, NaN for reference-set compounds.
        - max_sim_{cutoff_year}: float, NaN for reference-set compounds.
        - most_sim_cpd_pre_{cutoff_year}: str, NaN for reference-set compounds.
    """
    col_novel = f"is_novel_{cutoff_year}"
    col_sim = f"max_sim_pre_{cutoff_year}"
    col_most_sim = f"most_sim_cpd_pre_{cutoff_year}"

    indexed = compounds_df.set_index("chembl_id")

    result = pd.DataFrame(
        {col_novel: pd.NA, col_sim: np.nan, col_most_sim: pd.NA},
        index=indexed.index,
    )
    result.index.name = "chembl_id"

    if threshold is None:
        result[col_novel] = True
        return result

    ref_ids = np.array(
        [
            cid
            for cid in indexed.index
            if cid in fp_index and indexed.loc[cid, "cpd_earliest_year"] < cutoff_year
        ]
    )
    ref_fps = (
        fp_matrix[[fp_index[cid] for cid in ref_ids]]
        if len(ref_ids) > 0
        else np.empty((0, fp_matrix.shape[1]))
    )

    cand_ids = np.array(
        [
            cid
            for cid in indexed.index
            if cid in fp_index and indexed.loc[cid, "cpd_earliest_year"] >= cutoff_year
        ]
    )

    if len(cand_ids) == 0:
        return result

    cand_fps = fp_matrix[[fp_index[cid] for cid in cand_ids]]

    is_novel, max_sims, most_similar_ids = filter_by_tanimoto(
        candidate_fps=cand_fps,
        reference_fps=ref_fps,
        reference_ids=ref_ids,
        threshold=threshold,
        n_jobs=n_jobs,
    )

    result.loc[cand_ids, col_novel] = is_novel
    result.loc[cand_ids, col_sim] = max_sims
    result.loc[cand_ids, col_most_sim] = most_similar_ids

    return result
