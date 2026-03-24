import numpy as np
import pytest

from timesplit_affinity_benchmark.tanimoto_filter import filter_by_tanimoto


def _make_fp(on_bits: list[int], size: int = 2048) -> np.ndarray:
    """Create a binary fingerprint with specified bit positions set to 1."""
    fp = np.zeros(size, dtype=np.uint8)
    fp[on_bits] = 1
    return fp


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
        is_novel, max_sims, _ = filter_by_tanimoto(candidates, references, ref_ids, threshold=0.6)
        assert is_novel.tolist() == [True]
        assert pytest.approx(max_sims[0]) == 0.5

        # threshold exactly 0.5 → NOT novel (need strictly <)
        is_novel, max_sims, _ = filter_by_tanimoto(candidates, references, ref_ids, threshold=0.5)
        assert is_novel.tolist() == [False]
        assert pytest.approx(max_sims[0]) == 0.5

        # threshold below 0.5 → not novel
        is_novel, _, _ = filter_by_tanimoto(candidates, references, ref_ids, threshold=0.4)
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
        fp_disjoint = _make_fp([100, 101, 102])    # sim = 0.0 → novel
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
        fp_candidate = _make_fp(list(range(10)))       # bits 0-9
        fp_similar = _make_fp(list(range(5)))           # bits 0-4 → Tanimoto = 0.5
        fp_dissimilar = _make_fp(list(range(50, 60)))   # no overlap → Tanimoto = 0.0
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
