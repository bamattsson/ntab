import numpy as np
import pandas as pd
import pytest

from timesplit_affinity_benchmark.novelty import compute_novelty_for_cutoff


def _make_fp(on_bits: list[int], size: int = 2048) -> np.ndarray:
    """Create a binary fingerprint with specified bit positions set to 1."""
    fp = np.zeros(size, dtype=np.uint8)
    fp[on_bits] = 1
    return fp


def _make_fixtures() -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Shared minimal fixture set.

    Compounds:
    - "A": cpd_earliest_year=2020 (train),          fp = bits 0-4
    - "B": cpd_earliest_year=2025 (novel test),      fp = bits 100-104 (disjoint from A)
    - "C": cpd_earliest_year=2025 (not-novel test),  fp = bits 0-4 (identical to A)
    - "D": cpd_earliest_year=2023 (val),             fp = bits 200-202 (disjoint from A)
    """
    compounds = pd.DataFrame(
        {
            "chembl_id": ["A", "B", "C", "D"],
            "cpd_earliest_year": [2020, 2025, 2025, 2023],
        }
    )
    fp_matrix = np.array(
        [
            _make_fp([0, 1, 2, 3, 4]),  # A
            _make_fp([100, 101, 102, 103, 104]),  # B — disjoint from A
            _make_fp([0, 1, 2, 3, 4]),  # C — identical to A
            _make_fp([200, 201, 202]),  # D — disjoint from A
        ]
    )
    fp_index = {"A": 0, "B": 1, "C": 2, "D": 3}
    return compounds, fp_matrix, fp_index


class TestComputeNoveltyForCutoff:
    """compute_novelty_for_cutoff returns compounds_df enriched with 3 novelty columns.

    Reference compounds (cpd_earliest_year < cutoff_year) are pre-filtered and receive
    NaN for all three columns.  Candidates (cpd_earliest_year >= cutoff_year) receive
    is_novel_{year}, max_sim_{year}, most_similar_id_{year} from filter_by_tanimoto.
    """

    def test_2024_cutoff_structure_and_values(self) -> None:
        """Output has correct columns, index, NaN for reference, and novelty values for candidates."""
        compounds, fp_matrix, fp_index = _make_fixtures()
        result = compute_novelty_for_cutoff(
            compounds_df=compounds,
            cutoff_year=2024,
            fp_index=fp_index,
            fp_matrix=fp_matrix,
            threshold=0.35,
        )
        # Columns and index
        assert {"is_novel_2024", "max_sim_pre_2024", "most_sim_cpd_pre_2024"}.issubset(
            result.columns
        )
        assert result.index.name == "chembl_id"
        assert set(result.index) == {"A", "B", "C", "D"}

        # A is reference → all NaN
        assert pd.isna(result.loc["A", "is_novel_2024"])
        assert pd.isna(result.loc["A", "max_sim_pre_2024"])
        assert pd.isna(result.loc["A", "most_sim_cpd_pre_2024"])

        # B is disjoint from reference → novel
        assert result.loc["B", "is_novel_2024"] == True
        assert pytest.approx(result.loc["B", "max_sim_pre_2024"]) == 0.0

        # C is identical to A → not novel, most similar is A
        assert result.loc["C", "is_novel_2024"] == False
        assert pytest.approx(result.loc["C", "max_sim_pre_2024"]) == 1.0
        assert result.loc["C", "most_sim_cpd_pre_2024"] == "A"

    def test_compound_in_both_splits_is_pre_filtered(self) -> None:
        """A compound with cpd_earliest_year < cutoff belongs to the reference set and
        receives NaN even if it also has activities after the cutoff."""
        compounds = pd.DataFrame({"chembl_id": ["A"], "cpd_earliest_year": [2020]})
        fp_matrix = np.array([_make_fp([0, 1, 2])])
        fp_index = {"A": 0}
        result = compute_novelty_for_cutoff(
            compounds_df=compounds,
            cutoff_year=2024,
            fp_index=fp_index,
            fp_matrix=fp_matrix,
            threshold=0.35,
        )
        assert pd.isna(result.loc["A", "is_novel_2024"])

    def test_threshold_none_marks_all_compounds_novel(self) -> None:
        """threshold=None skips Tanimoto computation; every compound (including
        reference-set compounds) is marked novel so nothing ends up as discard_not_novel."""
        compounds, fp_matrix, fp_index = _make_fixtures()
        result = compute_novelty_for_cutoff(
            compounds_df=compounds,
            cutoff_year=2024,
            fp_index=fp_index,
            fp_matrix=fp_matrix,
            threshold=None,
        )
        # All compounds are marked novel, including reference-set A
        for cid in ["A", "B", "C"]:
            assert result.loc[cid, "is_novel_2024"] == True
        # max_sim and most_sim columns remain NaN (not computed)
        for cid in ["A", "B", "C"]:
            assert pd.isna(result.loc[cid, "max_sim_pre_2024"])
            assert pd.isna(result.loc[cid, "most_sim_cpd_pre_2024"])

    def test_cutoff_year_2023_val_compounds_are_candidates(self) -> None:
        """With cutoff_year=2023, val-year (2023) and test-year (2025) compounds
        are all candidates; only pre-2023 compounds are reference."""
        compounds, fp_matrix, fp_index = _make_fixtures()
        result = compute_novelty_for_cutoff(
            compounds_df=compounds,
            cutoff_year=2023,
            fp_index=fp_index,
            fp_matrix=fp_matrix,
            threshold=0.35,
        )
        assert pd.isna(result.loc["A", "is_novel_2023"])  # reference → NaN
        assert result.loc["D", "is_novel_2023"] == True  # disjoint from A → novel
        assert result.loc["B", "is_novel_2023"] == True  # disjoint from A → novel
        assert result.loc["C", "is_novel_2023"] == False  # identical to A → not novel
