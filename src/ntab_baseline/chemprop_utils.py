"""Chemprop utilities: SMILES-to-MolGraph conversion, caching, and batch collation."""

from __future__ import annotations

import torch
from chemprop.data import BatchMolGraph, MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem


def get_featurizer() -> SimpleMoleculeMolGraphFeaturizer:
    """Return the default Chemprop molecular graph featurizer."""
    return SimpleMoleculeMolGraphFeaturizer()


class MolGraphCache:
    """Converts SMILES to MolGraph with in-memory caching.

    Args:
        featurizer: Chemprop featurizer. Uses the default if not provided.
    """

    def __init__(
        self, featurizer: SimpleMoleculeMolGraphFeaturizer | None = None
    ) -> None:
        self._featurizer = featurizer or get_featurizer()
        self._cache: dict[str, MolGraph] = {}

    def __call__(self, smiles: str) -> MolGraph:
        if smiles not in self._cache:
            mol = Chem.MolFromSmiles(smiles)
            self._cache[smiles] = self._featurizer(mol)
        return self._cache[smiles]


def collate_batch(batch: list[tuple]) -> tuple:
    """Collate a list of 7-element dataset samples into a batched tuple.

    Elements 0-4 (tensors) are stacked. Element 5 (assay_ids) is collected
    into a list. Element 6 (MolGraph or None) is collated into a BatchMolGraph
    or left as None.
    """
    fps = torch.stack([s[0] for s in batch])
    mol_props = torch.stack([s[1] for s in batch])
    target_idx = torch.stack([s[2] for s in batch])
    std_type_idx = torch.stack([s[3] for s in batch])
    labels = torch.stack([s[4] for s in batch])
    assay_ids = [s[5] for s in batch]

    mol_graphs = [s[6] for s in batch]
    if mol_graphs[0] is None:
        bmg = None
    else:
        bmg = BatchMolGraph(mol_graphs)

    return fps, mol_props, target_idx, std_type_idx, labels, assay_ids, bmg
