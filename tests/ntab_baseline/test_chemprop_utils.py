"""Tests for Chemprop utility functions."""

import torch
import pytest
from chemprop.data import BatchMolGraph, MolGraph

from ntab_baseline.chemprop_utils import (
    MolGraphCache,
    collate_batch,
    get_featurizer,
)

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class TestGetFeaturizer:
    def test_returns_featurizer(self):
        f = get_featurizer()
        assert hasattr(f, "atom_fdim")
        assert hasattr(f, "bond_fdim")

    def test_atom_and_bond_dims_are_positive(self):
        f = get_featurizer()
        assert f.atom_fdim > 0
        assert f.bond_fdim > 0


class TestMolGraphCache:
    def test_returns_molgraph(self):
        cache = MolGraphCache()
        mg = cache(ETHANOL)
        assert isinstance(mg, MolGraph)

    def test_caches_result(self):
        cache = MolGraphCache()
        mg1 = cache(ETHANOL)
        mg2 = cache(ETHANOL)
        assert mg1 is mg2

    def test_different_smiles_different_graphs(self):
        cache = MolGraphCache()
        mg1 = cache(ETHANOL)
        mg2 = cache(BENZENE)
        assert mg1 is not mg2

    def test_custom_featurizer(self):
        f = get_featurizer()
        cache = MolGraphCache(featurizer=f)
        mg = cache(ASPIRIN)
        assert isinstance(mg, MolGraph)

    def test_molgraph_has_atoms_and_bonds(self):
        cache = MolGraphCache()
        mg = cache(ETHANOL)
        assert mg.V.shape[0] > 0
        assert mg.E.shape[0] > 0


class TestCollateBatch:
    def _make_sample(self, smiles: str, cache: MolGraphCache):
        """Create a 7-element tuple mimicking AffinityDataset.__getitem__."""
        fp = torch.randn(2048)
        mol_props = torch.randn(12)
        target_idx = torch.tensor(0)
        std_type_idx = torch.tensor(1)
        label = torch.tensor(5.5)
        assay_id = "CHEMBL123_IC50"
        mol_graph = cache(smiles)
        return (fp, mol_props, target_idx, std_type_idx, label, assay_id, mol_graph)

    def test_returns_correct_number_of_elements(self):
        cache = MolGraphCache()
        batch = [self._make_sample(ETHANOL, cache), self._make_sample(BENZENE, cache)]
        result = collate_batch(batch)
        assert len(result) == 7

    def test_tensors_are_stacked(self):
        cache = MolGraphCache()
        batch = [self._make_sample(ETHANOL, cache), self._make_sample(BENZENE, cache)]
        fps, mol_props, target_idx, std_type_idx, labels, assay_ids, bmg = (
            collate_batch(batch)
        )
        assert fps.shape == (2, 2048)
        assert mol_props.shape == (2, 12)
        assert target_idx.shape == (2,)
        assert std_type_idx.shape == (2,)
        assert labels.shape == (2,)

    def test_assay_ids_are_list(self):
        cache = MolGraphCache()
        batch = [self._make_sample(ETHANOL, cache), self._make_sample(BENZENE, cache)]
        result = collate_batch(batch)
        assay_ids = result[5]
        assert isinstance(assay_ids, list)
        assert len(assay_ids) == 2

    def test_bmg_is_batch_mol_graph(self):
        cache = MolGraphCache()
        batch = [self._make_sample(ETHANOL, cache), self._make_sample(BENZENE, cache)]
        result = collate_batch(batch)
        bmg = result[6]
        assert isinstance(bmg, BatchMolGraph)

    def test_single_sample_batch(self):
        cache = MolGraphCache()
        batch = [self._make_sample(ASPIRIN, cache)]
        fps, mol_props, target_idx, std_type_idx, labels, assay_ids, bmg = (
            collate_batch(batch)
        )
        assert fps.shape == (1, 2048)
        assert isinstance(bmg, BatchMolGraph)

    def test_none_molgraph_passes_through_as_none(self):
        """When molgraph is None (chemprop disabled), collate should return None."""
        fp = torch.randn(2048)
        mol_props = torch.randn(12)
        sample = (fp, mol_props, torch.tensor(0), torch.tensor(1), torch.tensor(5.0), "A", None)
        batch = [sample, sample]
        result = collate_batch(batch)
        assert result[6] is None
