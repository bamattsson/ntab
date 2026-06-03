import numpy as np

from ntab_baseline.preprocess_utils import compute_fingerprints

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL_SMILES = "CCO"
ETHANOL_WITH_NACL_SMILES = "CCO.[Na+].[Cl-]"  # ethanol with inorganic salt spectators
INVALID_SMILES = "this-is-not-a-smiles"


class TestComputeFingerprints:
    def test_binary_output_contains_only_zero_and_one(self) -> None:
        _, fps = compute_fingerprints(["mol"], [ASPIRIN_SMILES], fp_type="binary")
        assert set(fps[0].tolist()).issubset({0, 1})

    def test_count_output_contains_non_negative_integers(self) -> None:
        _, fps = compute_fingerprints(["mol"], [ASPIRIN_SMILES], fp_type="count")
        assert (fps[0] >= 0).all()
        assert fps[0].dtype.kind in ("u", "i")  # unsigned or signed integer

    def test_count_values_ge_binary_values_elementwise(self) -> None:
        _, binary_fps = compute_fingerprints(
            ["mol"], [ASPIRIN_SMILES], fp_type="binary"
        )
        _, count_fps = compute_fingerprints(["mol"], [ASPIRIN_SMILES], fp_type="count")
        # Wherever the binary bit is set, the count must be at least 1
        assert (count_fps[0] >= binary_fps[0]).all()

    def test_valid_smiles_produces_correct_output_shape(self) -> None:
        fp_size = 512
        _, fps = compute_fingerprints(["mol"], [ASPIRIN_SMILES], fp_size=fp_size)
        assert fps.shape == (1, fp_size)

    def test_multiple_molecules_output_shape(self) -> None:
        names, fps = compute_fingerprints(
            ["a", "b"], [ASPIRIN_SMILES, ETHANOL_SMILES], fp_type="binary"
        )
        assert fps.shape == (2, 2048)
        assert len(names) == 2

    def test_invalid_smiles_is_skipped_no_crash(self) -> None:
        names, fps = compute_fingerprints(
            ["valid", "invalid"],
            [ASPIRIN_SMILES, INVALID_SMILES],
            fp_type="binary",
        )
        assert len(names) == 1
        assert names[0] == "valid"
        assert fps.shape == (1, 2048)

    def test_all_invalid_smiles_returns_empty_arrays(self) -> None:
        names, fps = compute_fingerprints(["bad"], [INVALID_SMILES], fp_type="binary")
        assert len(names) == 0
        assert len(fps) == 0

    def test_binary_is_deterministic(self) -> None:
        _, fps1 = compute_fingerprints(["mol"], [ASPIRIN_SMILES], fp_type="binary")
        _, fps2 = compute_fingerprints(["mol"], [ASPIRIN_SMILES], fp_type="binary")
        np.testing.assert_array_equal(fps1, fps2)

    def test_count_is_deterministic(self) -> None:
        _, fps1 = compute_fingerprints(["mol"], [ASPIRIN_SMILES], fp_type="count")
        _, fps2 = compute_fingerprints(["mol"], [ASPIRIN_SMILES], fp_type="count")
        np.testing.assert_array_equal(fps1, fps2)

    def test_names_preserved_in_output_order(self) -> None:
        names, _ = compute_fingerprints(
            ["aspirin", "ethanol"],
            [ASPIRIN_SMILES, ETHANOL_SMILES],
        )
        assert list(names) == ["aspirin", "ethanol"]

    def test_names_exclude_failed_molecules(self) -> None:
        names, _ = compute_fingerprints(
            ["aspirin", "bad", "ethanol"],
            [ASPIRIN_SMILES, INVALID_SMILES, ETHANOL_SMILES],
        )
        assert list(names) == ["aspirin", "ethanol"]

    def test_default_fp_size_is_2048(self) -> None:
        _, fps = compute_fingerprints(["mol"], [ASPIRIN_SMILES])
        assert fps.shape[1] == 2048

    def test_inorganic_salt_spectators_stripped(self) -> None:
        _, fps_pure = compute_fingerprints(["pure"], [ETHANOL_SMILES])
        _, fps_salt = compute_fingerprints(["salt"], [ETHANOL_WITH_NACL_SMILES])
        np.testing.assert_array_equal(fps_pure[0], fps_salt[0])

    def test_parallel_matches_serial(self) -> None:
        mol_names = ["aspirin", "ethanol"]
        smiles = [ASPIRIN_SMILES, ETHANOL_SMILES]
        _, fps_serial = compute_fingerprints(
            mol_names, smiles, fp_type="binary", n_jobs=1
        )
        _, fps_parallel = compute_fingerprints(
            mol_names, smiles, fp_type="binary", n_jobs=2
        )
        np.testing.assert_array_equal(fps_serial, fps_parallel)
