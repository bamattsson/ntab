"""PyTorch Dataset for the binding prediction baseline."""

import numpy as np
import torch
from torch.utils.data import Dataset


class AffinityDataset(Dataset):
    """Dataset of (fingerprint, props, target_index, standard_type_index, pchembl_label, assay_id) tuples.

    Fingerprints and mol properties are looked up from shared pre-loaded matrices
    using per-sample indices — no data is copied or stored per-sample.

    Args:
        fps_matrix: Full fingerprint matrix from fingerprints_ecfp4.npz, shape
            (N_total, fp_size), float32 tensor. Shared across train/val/test.
        props_matrix: Full mol properties matrix from mol_properties.npz, shape
            (N_total, n_props), float32 tensor. Shared across train/val/test.
        fp_indices: Row index per sample into fps_matrix and props_matrix, length N.
        target_indices: Integer target index per sample, length N.
        standard_type_indices: Integer standard type index per sample, length N.
        labels: pchembl_value_filled per sample, length N.
        assay_ids: Assay identifier string per sample, length N.
    """

    def __init__(
        self,
        fps_matrix: torch.Tensor,
        props_matrix: torch.Tensor,
        fp_indices: list[int] | np.ndarray,
        target_indices: list[int] | np.ndarray,
        standard_type_indices: list[int] | np.ndarray,
        labels: np.ndarray,
        assay_ids: list[str],
    ) -> None:
        self._fps = fps_matrix
        self._mol_props = props_matrix
        self._fp_indices = torch.tensor(fp_indices, dtype=torch.long)
        self._target_indices = torch.tensor(target_indices, dtype=torch.long)
        self._standard_type_indices = torch.tensor(standard_type_indices, dtype=torch.long)
        self._labels = torch.tensor(labels, dtype=torch.float32)
        self._assay_ids = list(assay_ids)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        fp = self._fps[self._fp_indices[idx]]
        mol_props = self._mol_props[self._fp_indices[idx]]
        return (
            fp,
            mol_props,
            self._target_indices[idx],
            self._standard_type_indices[idx],
            self._labels[idx],
            self._assay_ids[idx],
        )
