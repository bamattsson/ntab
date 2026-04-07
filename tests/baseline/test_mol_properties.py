"""Tests for mol_properties.compute_mol_properties and normalise_mol_properties."""

import numpy as np
import pytest

from bind_pred_baseline.preprocess_utils import (
    FEATURE_NAMES,
    compute_mol_properties,
    normalise_mol_properties,
)

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL_SMILES = "CCO"
ETHANOL_WITH_NACL_SMILES = "CCO.[Na+].[Cl-]"
INVALID_SMILES = "this-is-not-a-smiles"

# ---------------------------------------------------------------------------
# FEATURE_NAMES
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_exactly_12_features(self) -> None:
        assert len(FEATURE_NAMES) == 12

    def test_all_feature_names_are_strings(self) -> None:
        assert all(isinstance(n, str) for n in FEATURE_NAMES)

    def test_no_duplicate_feature_names(self) -> None:
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_tier1_features_present(self) -> None:
        # These six must be in FEATURE_NAMES (exact spelling may differ but key words must match)
        tier1 = [
            "MolLogP",
            "ExactMolWt",
            "TPSA",
            "NumHDonors",
            "NumHAcceptors",
            "NumRotatableBonds",
        ]
        for name in tier1:
            assert name in FEATURE_NAMES, f"{name!r} not found in FEATURE_NAMES"

    def test_tier2_features_present(self) -> None:
        tier2 = [
            "FormalCharge",
            "MolMR",
            "FractionCSP3",
            "RingCount",
            "NumAromaticRings",
            "HeavyAtomCount",
        ]
        for name in tier2:
            assert name in FEATURE_NAMES, f"{name!r} not found in FEATURE_NAMES"


# ---------------------------------------------------------------------------
# compute_mol_properties — output shape and types
# ---------------------------------------------------------------------------


class TestComputeMolPropertiesShape:
    def test_single_valid_molecule_shape(self) -> None:
        names, props = compute_mol_properties(["aspirin"], [ASPIRIN_SMILES])
        assert props.shape == (1, 12)

    def test_multiple_valid_molecules_shape(self) -> None:
        names, props = compute_mol_properties(
            ["aspirin", "ethanol"], [ASPIRIN_SMILES, ETHANOL_SMILES]
        )
        assert props.shape == (2, 12)

    def test_output_dtype_is_float32(self) -> None:
        _, props = compute_mol_properties(["aspirin"], [ASPIRIN_SMILES])
        assert props.dtype == np.float32

    def test_names_returned_as_numpy_array(self) -> None:
        names, _ = compute_mol_properties(["aspirin"], [ASPIRIN_SMILES])
        assert isinstance(names, np.ndarray)


# ---------------------------------------------------------------------------
# compute_mol_properties — known values for aspirin
# ---------------------------------------------------------------------------


class TestKnownValuesAspirin:
    """Spot-check a subset of features against published / RDKit-verified values."""

    @pytest.fixture(autouse=True)
    def _compute(self) -> None:
        names, props = compute_mol_properties(["aspirin"], [ASPIRIN_SMILES])
        self.row = dict(zip(FEATURE_NAMES, props[0].tolist()))

    def test_h_bond_donors(self) -> None:
        # Aspirin has one OH (COOH), so NumHDonors = 1
        assert self.row["NumHDonors"] == pytest.approx(1.0)

    def test_h_bond_acceptors(self) -> None:
        # RDKit uses the Ertl definition: excludes OH/NH (those are donors).
        # Aspirin has 3 acceptor oxygens: ester C=O, ester bridge O, and COOH C=O.
        # The COOH OH is a donor, not counted here.
        assert self.row["NumHAcceptors"] == pytest.approx(3.0)

    def test_ring_count(self) -> None:
        assert self.row["RingCount"] == pytest.approx(1.0)

    def test_aromatic_rings(self) -> None:
        assert self.row["NumAromaticRings"] == pytest.approx(1.0)

    def test_formal_charge(self) -> None:
        assert self.row["FormalCharge"] == pytest.approx(0.0)

    def test_heavy_atom_count(self) -> None:
        # Aspirin C9H8O4: 9 C + 4 O = 13 heavy atoms
        assert self.row["HeavyAtomCount"] == pytest.approx(13.0)

    def test_mol_logp_reasonable_range(self) -> None:
        # Crippen logP for aspirin ≈ 1.31
        assert 1.0 <= self.row["MolLogP"] <= 2.0

    def test_exact_mol_wt_reasonable_range(self) -> None:
        # Exact MW of aspirin = 180.042
        assert 179.0 <= self.row["ExactMolWt"] <= 181.0

    def test_tpsa_reasonable_range(self) -> None:
        # TPSA of aspirin ≈ 63.60 Å²
        assert 60.0 <= self.row["TPSA"] <= 70.0

    def test_fraction_csp3_reasonable(self) -> None:
        # Aspirin has 9 C atoms, only the methyl is sp3 → FractionCSP3 ≈ 1/9 ≈ 0.111
        assert 0.05 <= self.row["FractionCSP3"] <= 0.20


# ---------------------------------------------------------------------------
# compute_mol_properties — invalid / edge-case SMILES
# ---------------------------------------------------------------------------


class TestComputeMolPropertiesEdgeCases:
    def test_invalid_smiles_skipped_no_crash(self) -> None:
        names, props = compute_mol_properties(
            ["valid", "invalid"], [ASPIRIN_SMILES, INVALID_SMILES]
        )
        assert len(names) == 1
        assert names[0] == "valid"
        assert props.shape == (1, 12)

    def test_all_invalid_returns_empty(self) -> None:
        names, props = compute_mol_properties(["bad"], [INVALID_SMILES])
        assert len(names) == 0
        assert props.shape == (0, 12)

    def test_names_preserved_in_order(self) -> None:
        names, _ = compute_mol_properties(
            ["aspirin", "ethanol"], [ASPIRIN_SMILES, ETHANOL_SMILES]
        )
        assert list(names) == ["aspirin", "ethanol"]

    def test_names_exclude_failed_molecules(self) -> None:
        names, _ = compute_mol_properties(
            ["aspirin", "bad", "ethanol"],
            [ASPIRIN_SMILES, INVALID_SMILES, ETHANOL_SMILES],
        )
        assert list(names) == ["aspirin", "ethanol"]

    def test_salt_spectators_stripped(self) -> None:
        """Ethanol with NaCl spectators should produce identical features to pure ethanol."""
        _, props_pure = compute_mol_properties(["pure"], [ETHANOL_SMILES])
        _, props_salt = compute_mol_properties(["salt"], [ETHANOL_WITH_NACL_SMILES])
        np.testing.assert_array_almost_equal(props_pure[0], props_salt[0], decimal=4)

    def test_deterministic(self) -> None:
        _, props1 = compute_mol_properties(["aspirin"], [ASPIRIN_SMILES])
        _, props2 = compute_mol_properties(["aspirin"], [ASPIRIN_SMILES])
        np.testing.assert_array_equal(props1, props2)

    def test_parallel_matches_serial(self) -> None:
        mol_names = ["aspirin", "ethanol"]
        smiles = [ASPIRIN_SMILES, ETHANOL_SMILES]
        _, props_serial = compute_mol_properties(mol_names, smiles, n_jobs=1)
        _, props_parallel = compute_mol_properties(mol_names, smiles, n_jobs=2)
        np.testing.assert_array_equal(props_serial, props_parallel)


# ---------------------------------------------------------------------------
# normalise_mol_properties
# ---------------------------------------------------------------------------


class TestNormaliseMolProperties:
    def _make_props(self) -> np.ndarray:
        """Return raw properties for aspirin + ethanol as a (2, 12) float32 array."""
        _, props = compute_mol_properties(
            ["aspirin", "ethanol"], [ASPIRIN_SMILES, ETHANOL_SMILES]
        )
        return props

    def test_normalised_shape_preserved(self) -> None:
        props = self._make_props()
        normed = normalise_mol_properties(props, props.mean(axis=0), props.std(axis=0))
        assert normed.shape == props.shape

    def test_output_dtype_is_float32(self) -> None:
        props = self._make_props()
        normed = normalise_mol_properties(props, props.mean(axis=0), props.std(axis=0))
        assert normed.dtype == np.float32

    def test_self_normalised_mean_is_near_zero(self) -> None:
        _, props = compute_mol_properties(
            ["aspirin", "ethanol", "aspirin", "ethanol"],
            [ASPIRIN_SMILES, ETHANOL_SMILES, ASPIRIN_SMILES, ETHANOL_SMILES],
        )
        normed = normalise_mol_properties(props, props.mean(axis=0), props.std(axis=0))
        non_const = normed.std(axis=0) > 0
        if non_const.any():
            np.testing.assert_allclose(
                normed[:, non_const].mean(axis=0), 0.0, atol=1e-5
            )

    def test_self_normalised_std_is_near_one(self) -> None:
        _, props = compute_mol_properties(
            ["aspirin", "ethanol", "aspirin", "ethanol"],
            [ASPIRIN_SMILES, ETHANOL_SMILES, ASPIRIN_SMILES, ETHANOL_SMILES],
        )
        normed = normalise_mol_properties(props, props.mean(axis=0), props.std(axis=0))
        non_const = normed.std(axis=0) > 1e-6
        if non_const.any():
            np.testing.assert_allclose(normed[:, non_const].std(axis=0), 1.0, atol=1e-4)

    def test_constant_column_becomes_zero_not_nan(self) -> None:
        props = np.ones((3, 12), dtype=np.float32)
        mean = props.mean(axis=0)
        std = props.std(axis=0)
        normed = normalise_mol_properties(props, mean, std)
        assert not np.isnan(normed).any()
        np.testing.assert_array_equal(normed, np.zeros((3, 12), dtype=np.float32))

    def test_external_mean_std_used_when_provided(self) -> None:
        """mean=0, std=1 → normalised should equal raw."""
        props = self._make_props()
        mean = np.zeros(12, dtype=np.float32)
        std = np.ones(12, dtype=np.float32)
        normed = normalise_mol_properties(props, mean=mean, std=std)
        np.testing.assert_array_almost_equal(normed, props)

    def test_train_subset_normalisation_applies_train_scale_to_all(self) -> None:
        """Simulate train-only normalisation: fit on train rows, apply to all."""
        _, props = compute_mol_properties(
            ["aspirin", "ethanol"], [ASPIRIN_SMILES, ETHANOL_SMILES]
        )
        train_mean = props[:1].mean(axis=0)
        train_std = props[:1].std(axis=0)
        normed = normalise_mol_properties(props, mean=train_mean, std=train_std)
        assert normed.shape == (2, 12)
