"""Chemprop utilities: SMILES-to-MolGraph conversion and batch collation."""

from __future__ import annotations

import torch
from chemprop.data import BatchMolGraph, MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem
from tqdm import tqdm


def get_featurizer() -> SimpleMoleculeMolGraphFeaturizer:
    """Return the default Chemprop molecular graph featurizer."""
    return SimpleMoleculeMolGraphFeaturizer()


def smiles_to_molgraph(
    smiles: str,
    featurizer: SimpleMoleculeMolGraphFeaturizer | None = None,
) -> MolGraph:
    """Convert a SMILES string to a Chemprop MolGraph.

    Args:
        smiles: SMILES string.
        featurizer: Chemprop featurizer. Uses the default if not provided.

    Returns:
        MolGraph NamedTuple with atom features, bond features, and edge indices.

    Raises:
        ValueError: If RDKit cannot parse the SMILES.
    """
    feat = featurizer or get_featurizer()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return feat(mol)


def precompute_molgraphs(
    smiles_list: list[str],
    featurizer: SimpleMoleculeMolGraphFeaturizer | None = None,
) -> list[MolGraph | None]:
    """Featurize a list of SMILES into MolGraph objects.

    Args:
        smiles_list: SMILES strings, one per unique compound.
        featurizer: Chemprop featurizer. Uses the default if not provided.

    Returns:
        List of MolGraph objects (or None for unparseable SMILES), same order
        and indexing as the input list.
    """
    feat = featurizer or get_featurizer()
    results: list[MolGraph | None] = []
    for smi in tqdm(smiles_list, desc="Featurizing MolGraphs"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(None)
        else:
            results.append(feat(mol))
    return results


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
