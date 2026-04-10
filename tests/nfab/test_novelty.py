import numpy as np
import pandas as pd
import pytest

from nfab.novelty import (
    compute_ecfp4_fingerprints,
    compute_novelty_for_cutoff,
    filter_by_tanimoto,
)


def _make_fp(on_bits: list[int], size: int = 2048) -> np.ndarray:
    """Create a binary fingerprint with specified bit positions set to 1."""
    fp = np.zeros(size, dtype=np.uint8)
    fp[on_bits] = 1
    return fp


# ---------------------------------------------------------------------------
# compute_novelty_for_cutoff
# ---------------------------------------------------------------------------


def _make_novelty_fixtures() -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
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
        compounds, fp_matrix, fp_index = _make_novelty_fixtures()
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
        compounds, fp_matrix, fp_index = _make_novelty_fixtures()
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
        compounds, fp_matrix, fp_index = _make_novelty_fixtures()
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


# ---------------------------------------------------------------------------
# filter_by_tanimoto
# ---------------------------------------------------------------------------


class TestFilterByTanimoto:
    """filter_by_tanimoto returns (is_novel, max_similarities, most_similar_ids).

    is_novel[i] is True iff max Tanimoto similarity of candidate i against all
    references is strictly less than threshold.
    """

    def test_identical_fps_not_novel(self) -> None:
        fp = _make_fp([0, 1, 2, 3, 4])
        is_novel, max_sims, most_similar_ids = filter_by_tanimoto(
            candidate_fps=np.array([fp]),
            reference_fps=np.array([fp]),
            reference_ids=np.array(["ref_0"]),
            threshold=0.35,
        )
        assert is_novel.tolist() == [False]
        assert pytest.approx(max_sims[0]) == 1.0
        assert most_similar_ids.tolist() == ["ref_0"]

    def test_disjoint_fps_novel(self) -> None:
        fp_a = _make_fp([0, 1, 2])
        fp_b = _make_fp([100, 101, 102])
        is_novel, max_sims, most_similar_ids = filter_by_tanimoto(
            candidate_fps=np.array([fp_a]),
            reference_fps=np.array([fp_b]),
            reference_ids=np.array(["ref_0"]),
            threshold=0.35,
        )
        assert is_novel.tolist() == [True]
        assert pytest.approx(max_sims[0]) == 0.0
        assert most_similar_ids.tolist() == ["ref_0"]

    def test_known_tanimoto_boundary(self) -> None:
        # fp_a has bits 0-9 (10 bits on), fp_b has bits 0-4 (5 bits on)
        # intersection = 5, union = 10 → Tanimoto = 5/10 = 0.5
        fp_a = _make_fp(list(range(10)))
        fp_b = _make_fp(list(range(5)))
        candidates = np.array([fp_a])
        references = np.array([fp_b])
        ref_ids = np.array(["ref_0"])

        # threshold strictly above 0.5 → novel
        is_novel, max_sims, _ = filter_by_tanimoto(
            candidates, references, ref_ids, threshold=0.6
        )
        assert is_novel.tolist() == [True]
        assert pytest.approx(max_sims[0]) == 0.5

        # threshold exactly 0.5 → NOT novel (need strictly <)
        is_novel, max_sims, _ = filter_by_tanimoto(
            candidates, references, ref_ids, threshold=0.5
        )
        assert is_novel.tolist() == [False]
        assert pytest.approx(max_sims[0]) == 0.5

        # threshold below 0.5 → not novel
        is_novel, _, _ = filter_by_tanimoto(
            candidates, references, ref_ids, threshold=0.4
        )
        assert is_novel.tolist() == [False]

    def test_empty_candidates(self) -> None:
        fp = _make_fp([0, 1, 2])
        is_novel, max_sims, most_similar_ids = filter_by_tanimoto(
            candidate_fps=np.empty((0, 2048), dtype=np.uint8),
            reference_fps=np.array([fp]),
            reference_ids=np.array(["ref_0"]),
            threshold=0.35,
        )
        assert is_novel.shape == (0,)
        assert max_sims.shape == (0,)
        assert most_similar_ids.shape == (0,)

    def test_empty_references(self) -> None:
        # No reference compounds → every candidate counts as novel, similarity = 0.0
        fp = _make_fp([0, 1, 2])
        is_novel, max_sims, most_similar_ids = filter_by_tanimoto(
            candidate_fps=np.array([fp]),
            reference_fps=np.empty((0, 2048), dtype=np.uint8),
            reference_ids=np.empty(0, dtype=str),
            threshold=0.35,
        )
        assert is_novel.tolist() == [True]
        assert pytest.approx(max_sims[0]) == 0.0
        assert most_similar_ids.tolist() == [None]

    def test_multiple_candidates_mixed(self) -> None:
        fp_ref = _make_fp([0, 1, 2, 3, 4])
        fp_identical = _make_fp([0, 1, 2, 3, 4])  # sim = 1.0 → not novel
        fp_disjoint = _make_fp([100, 101, 102])  # sim = 0.0 → novel
        is_novel, max_sims, most_similar_ids = filter_by_tanimoto(
            candidate_fps=np.array([fp_identical, fp_disjoint]),
            reference_fps=np.array([fp_ref]),
            reference_ids=np.array(["ref_0"]),
            threshold=0.35,
        )
        assert is_novel.tolist() == [False, True]
        assert pytest.approx(max_sims[0]) == 1.0
        assert pytest.approx(max_sims[1]) == 0.0
        assert most_similar_ids.tolist() == ["ref_0", "ref_0"]

    def test_takes_max_over_references(self) -> None:
        # Candidate is similar to one reference but not another —
        # must report the most similar one
        fp_candidate = _make_fp(list(range(10)))  # bits 0-9
        fp_similar = _make_fp(list(range(5)))  # bits 0-4 → Tanimoto = 0.5
        fp_dissimilar = _make_fp(list(range(50, 60)))  # no overlap → Tanimoto = 0.0
        is_novel, max_sims, most_similar_ids = filter_by_tanimoto(
            candidate_fps=np.array([fp_candidate]),
            reference_fps=np.array([fp_dissimilar, fp_similar]),
            reference_ids=np.array(["ref_dissimilar", "ref_similar"]),
            threshold=0.4,
        )
        assert is_novel.tolist() == [False]
        assert pytest.approx(max_sims[0]) == 0.5
        assert most_similar_ids.tolist() == ["ref_similar"]

    def test_multiprocessing_matches_single(self) -> None:
        rng = np.random.default_rng(42)
        candidates = (rng.random((20, 2048)) > 0.9).astype(np.uint8)
        references = (rng.random((10, 2048)) > 0.9).astype(np.uint8)
        ref_ids = np.array([f"ref_{i}" for i in range(10)])
        is_novel_s, max_sims_s, ids_s = filter_by_tanimoto(
            candidates, references, ref_ids, threshold=0.35, n_jobs=1
        )
        is_novel_m, max_sims_m, ids_m = filter_by_tanimoto(
            candidates, references, ref_ids, threshold=0.35, n_jobs=2
        )
        np.testing.assert_array_equal(is_novel_s, is_novel_m)
        np.testing.assert_array_almost_equal(max_sims_s, max_sims_m)
        np.testing.assert_array_equal(ids_s, ids_m)


# ---------------------------------------------------------------------------
# compute_ecfp4_fingerprints
# ---------------------------------------------------------------------------


@pytest.fixture
def expected_fingerprints(fixtures_dir) -> tuple[np.ndarray, np.ndarray]:
    """Pre-computed ECFP4 fingerprints for the 100-row ChEMBL sample."""
    data = np.load(fixtures_dir / "ecfp4_expected.npz")
    return data["names"], data["fps"]


def test_compute_ecfp4_fingerprints(chembl_smiles_sample: pd.DataFrame) -> None:
    names, fps = compute_ecfp4_fingerprints(
        mol_names=chembl_smiles_sample["chembl_id"].tolist(),
        smiles=chembl_smiles_sample["canonical_smiles"].tolist(),
    )
    assert isinstance(names, np.ndarray)
    assert isinstance(fps, np.ndarray)
    assert len(names) == fps.shape[0]
    assert fps.shape[1] == 2048


def test_multiprocessing_matches_single_ecfp4(
    chembl_smiles_sample: pd.DataFrame,
    expected_fingerprints: tuple[np.ndarray, np.ndarray],
) -> None:
    """Multiprocessing output must be bit-identical to the pre-computed single-process result."""
    exp_names, exp_fps = expected_fingerprints

    names_mp, fps_mp = compute_ecfp4_fingerprints(
        mol_names=chembl_smiles_sample["chembl_id"].tolist(),
        smiles=chembl_smiles_sample["canonical_smiles"].tolist(),
        n_jobs=2,
    )

    assert list(names_mp) == list(exp_names)
    assert np.array_equal(fps_mp, exp_fps)
