"""Lightning CLI entry point for training the binding prediction baseline.

Usage:
    uv run python -m ntab_baseline.train fit --config configs/baseline/train.yaml
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import lightning as L
from lightning.pytorch.cli import LightningCLI
import torch
from torch.utils.data import DataLoader

from ntab_baseline.chemprop_utils import collate_batch
from ntab_baseline.constants import MOL_PROP_FEATURES
from ntab_baseline.dataset import AffinityDataset
from ntab_baseline.model import AffinityModel


class AffinityDataModule(L.LightningDataModule):
    """Loads preprocessed .npz split files and serves DataLoaders.

    Fingerprints are loaded once from fingerprints_ecfp4.npz (inside data_dir)
    and shared across all splits as a single tensor — no duplication.

    Args:
        data_dir: Directory containing train.npz, val.npz, test.npz,
            fingerprints_ecfp4.npz, and meta.json (written by preprocess_training_data.py).
        batch_size: Training and evaluation batch size.
        num_workers: DataLoader worker processes.
        use_chemprop: Whether to load SMILES and build MolGraphs for Chemprop.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 4,
        use_chemprop: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_chemprop = use_chemprop

        meta = json.loads((self.data_dir / "meta.json").read_text())
        self.n_targets: int = meta["n_targets"]
        self.n_standard_types: int = meta.get("n_standard_types", 3)
        self.n_mol_props: int = len(MOL_PROP_FEATURES)

    def _load_split(
        self,
        name: str,
        fps_matrix: torch.Tensor,
        mol_props_matrix: torch.Tensor,
        mol_graphs: list | None = None,
    ) -> AffinityDataset:
        data = np.load(self.data_dir / f"{name}.npz", allow_pickle=True)
        return AffinityDataset(
            fps_matrix=fps_matrix,
            props_matrix=mol_props_matrix,
            fp_indices=data["fp_indices"],
            target_indices=data["target_indices"],
            standard_type_indices=data["standard_type_indices"],
            labels=data["labels"],
            assay_ids=data["assay_ids"].tolist(),
            mol_graphs=mol_graphs,
        )

    def setup(self, stage: str | None = None) -> None:
        print("Loading fingerprints matrix...")
        fps_np = np.load(self.data_dir / "fingerprints_ecfp4.npz")["fps"].astype(
            np.float32
        )
        fps_matrix = torch.from_numpy(fps_np)
        print(f"  Loaded: {fps_matrix.shape}, {fps_matrix.nbytes / 1e9:.2f} GB")

        print("Loading mol properties matrix...")
        mol_props_npz = np.load(self.data_dir / "mol_properties.npz")
        feature_names = list(mol_props_npz["feature_names"])
        col_indices = [feature_names.index(f) for f in MOL_PROP_FEATURES]
        mol_props_matrix = torch.from_numpy(
            mol_props_npz["props"][:, col_indices].astype(np.float32)
        )
        print(
            f"  Loaded: {mol_props_matrix.shape} ({len(MOL_PROP_FEATURES)} features: {MOL_PROP_FEATURES})"
        )

        mol_graphs = None
        if self.use_chemprop:
            mg_path = self.data_dir / "molgraphs.pkl"
            print(f"Loading precomputed MolGraphs from {mg_path}...")
            with open(mg_path, "rb") as f:
                mol_graphs = pickle.load(f)
            print(f"  Loaded {len(mol_graphs)} MolGraph objects")

        self._train_ds = self._load_split(
            "train", fps_matrix, mol_props_matrix, mol_graphs
        )
        self._val_ds = self._load_split(
            "val", fps_matrix, mol_props_matrix, mol_graphs
        )
        self._test_ds = self._load_split(
            "test", fps_matrix, mol_props_matrix, mol_graphs
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )


class AffinityCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments(
            "data.n_targets", "model.n_targets", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.n_standard_types", "model.n_standard_types", apply_on="instantiate"
        )


def cli_main() -> None:
    AffinityCLI(AffinityModel, AffinityDataModule)  # noqa: F841


if __name__ == "__main__":
    cli_main()
