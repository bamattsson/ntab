import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ntab_baseline.chemprop_utils import collate_batch
from ntab_baseline.dataset import AffinityDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PROPS = 12


def _make_fps_matrix(n_rows: int = 5) -> torch.Tensor:
    rng = np.random.default_rng(0)
    fps = (rng.random((n_rows, 2048)) > 0.9).astype(np.float32)
    return torch.from_numpy(fps)


def _make_props_matrix(n_rows: int = 5) -> torch.Tensor:
    rng = np.random.default_rng(1)
    return torch.from_numpy(rng.standard_normal((n_rows, N_PROPS)).astype(np.float32))


def _make_dataset(n: int = 4, n_unique: int = 3) -> AffinityDataset:
    fps_matrix = _make_fps_matrix(n_unique)
    props_matrix = _make_props_matrix(n_unique)
    fp_indices = [i % n_unique for i in range(n)]
    target_indices = list(range(n))
    standard_type_indices = [i % 3 for i in range(n)]
    rng = np.random.default_rng(0)
    labels = rng.random(n).astype(np.float32) * 5 + 4
    assay_ids = [f"ASSAY_{i % 2}" for i in range(n)]
    return AffinityDataset(
        fps_matrix,
        props_matrix,
        fp_indices,
        target_indices,
        standard_type_indices,
        labels,
        assay_ids,
    )


# ---------------------------------------------------------------------------
# AffinityDataset
# ---------------------------------------------------------------------------


class TestAffinityDataset:
    def test_len_matches_number_of_samples(self) -> None:
        ds = _make_dataset(n=7, n_unique=3)
        assert len(ds) == 7

    def test_getitem_returns_seven_elements(self) -> None:
        ds = _make_dataset()
        assert len(ds[0]) == 7

    def test_fingerprint_is_float32_tensor_of_correct_size(self) -> None:
        ds = _make_dataset()
        fp, _, _, _, _, _, _ = ds[0]
        assert isinstance(fp, torch.Tensor)
        assert fp.dtype == torch.float32
        assert fp.shape == (2048,)

    def test_props_is_float32_tensor_of_correct_size(self) -> None:
        ds = _make_dataset()
        _, props, _, _, _, _, _ = ds[0]
        assert isinstance(props, torch.Tensor)
        assert props.dtype == torch.float32
        assert props.shape == (N_PROPS,)

    def test_target_idx_is_long_tensor(self) -> None:
        ds = _make_dataset()
        _, _, target_idx, _, _, _, _ = ds[0]
        assert isinstance(target_idx, torch.Tensor)
        assert target_idx.dtype == torch.long

    def test_standard_type_idx_is_long_tensor(self) -> None:
        ds = _make_dataset()
        _, _, _, std_type_idx, _, _, _ = ds[0]
        assert isinstance(std_type_idx, torch.Tensor)
        assert std_type_idx.dtype == torch.long

    def test_standard_type_idx_values_match_input(self) -> None:
        fps_matrix = torch.zeros(1, 2048)
        props_matrix = torch.zeros(1, N_PROPS)
        labels = np.array([6.0, 7.0, 8.0], dtype=np.float32)
        ds = AffinityDataset(
            fps_matrix,
            props_matrix,
            [0, 0, 0],
            [0, 1, 2],
            [0, 1, 2],
            labels,
            ["A", "B", "C"],
        )
        _, _, _, std_type_idx, _, _, _ = ds[1]
        assert std_type_idx.item() == 1

    def test_label_is_scalar_float32_tensor(self) -> None:
        ds = _make_dataset()
        _, _, _, _, label, _, _ = ds[0]
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.float32
        assert label.shape == ()

    def test_assay_id_is_string(self) -> None:
        ds = _make_dataset()
        _, _, _, _, _, assay_id, _ = ds[0]
        assert isinstance(assay_id, str)

    def test_fp_indices_look_up_correct_row_from_shared_matrix(self) -> None:
        fps_matrix = torch.zeros(2, 2048)
        fps_matrix[0, 0] = 1.0  # row 0 marker
        fps_matrix[1, 100] = 1.0  # row 1 marker
        props_matrix = torch.zeros(2, N_PROPS)
        fp_indices = [1, 0]
        labels = np.array([6.0, 7.0], dtype=np.float32)
        ds = AffinityDataset(
            fps_matrix, props_matrix, fp_indices, [0, 1], [0, 0], labels, ["A", "B"]
        )

        fp0, _, _, _, _, _, _ = ds[0]
        fp1, _, _, _, _, _, _ = ds[1]
        assert fp0[100].item() == 1.0  # from row 1
        assert fp1[0].item() == 1.0  # from row 0

    def test_values_match_input_at_index(self) -> None:
        fps_matrix = torch.zeros(1, 2048)
        props_matrix = torch.zeros(1, N_PROPS)
        labels = np.array([6.0, 7.0, 8.0], dtype=np.float32)
        ds = AffinityDataset(
            fps_matrix,
            props_matrix,
            [0, 0, 0],
            [0, 1, 2],
            [0, 1, 2],
            labels,
            ["A", "B", "C"],
        )

        _, _, t_idx, std_type_idx, label, assay_id, _ = ds[1]
        assert t_idx.item() == 1
        assert std_type_idx.item() == 1
        assert pytest.approx(label.item()) == 7.0
        assert assay_id == "B"

    def test_fps_matrix_is_shared_not_copied(self) -> None:
        fps_matrix = _make_fps_matrix(3)
        props_matrix = _make_props_matrix(3)
        labels = np.zeros(3, dtype=np.float32)
        ds2 = AffinityDataset(
            fps_matrix,
            props_matrix,
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            labels,
            ["A", "B", "C"],
        )
        assert ds2._fps is fps_matrix

    def test_dataloader_batches_correctly(self) -> None:
        ds = _make_dataset(n=8, n_unique=3)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batch)
        (
            fps_batch,
            props_batch,
            target_batch,
            std_type_batch,
            label_batch,
            assay_batch,
            mg_batch,
        ) = next(iter(loader))

        assert fps_batch.shape == (4, 2048)
        assert props_batch.shape == (4, N_PROPS)
        assert target_batch.shape == (4,)
        assert std_type_batch.shape == (4,)
        assert label_batch.shape == (4,)
        assert len(assay_batch) == 4

    def test_dataloader_covers_all_samples(self) -> None:
        ds = _make_dataset(n=6, n_unique=2)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batch)
        total = sum(fps.shape[0] for fps, _, _, _, _, _, _ in loader)
        assert total == 6

    def test_mol_graph_is_none_without_smiles(self) -> None:
        ds = _make_dataset()
        *_, mol_graph = ds[0]
        assert mol_graph is None

    def test_mol_graph_returned_with_smiles_and_fn(self) -> None:
        from chemprop.data import MolGraph
        from ntab_baseline.chemprop_utils import MolGraphCache

        fps_matrix = _make_fps_matrix(2)
        props_matrix = _make_props_matrix(2)
        labels = np.array([6.0, 7.0], dtype=np.float32)
        smiles = ["CCO", "c1ccccc1"]
        cache = MolGraphCache()
        ds = AffinityDataset(
            fps_matrix, props_matrix, [0, 1], [0, 1], [0, 0],
            labels, ["A", "B"], smiles=smiles, mol_graph_fn=cache,
        )
        *_, mol_graph = ds[0]
        assert isinstance(mol_graph, MolGraph)

    def test_mol_graph_cached_across_calls(self) -> None:
        from ntab_baseline.chemprop_utils import MolGraphCache

        fps_matrix = _make_fps_matrix(2)
        props_matrix = _make_props_matrix(2)
        labels = np.array([6.0, 7.0], dtype=np.float32)
        smiles = ["CCO", "CCO"]
        cache = MolGraphCache()
        ds = AffinityDataset(
            fps_matrix, props_matrix, [0, 1], [0, 1], [0, 0],
            labels, ["A", "B"], smiles=smiles, mol_graph_fn=cache,
        )
        *_, mg1 = ds[0]
        *_, mg2 = ds[1]
        assert mg1 is mg2
