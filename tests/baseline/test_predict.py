"""Tests for prediction mode: preprocess_data_for_prediction, PredictDataset,
AffinityModel.predict_step, and PredictWriterCallback."""

import json
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from bind_pred_baseline.constants import FP_SIZE, MOL_PROP_FEATURES, N_MOL_PROP_FEATURES, STANDARD_TYPE_INDEX
from bind_pred_baseline.dataset import PredictDataset
from bind_pred_baseline.model import AffinityModel
from bind_pred_baseline.model_utils import PredictWriterCallback


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_preproc_dir(tmp_path: Path, n_targets: int = 5) -> Path:
    """Create a minimal data_dir with required files."""
    preproc_dir = tmp_path / "preproc"
    preproc_dir.mkdir()

    # target_index.json
    target_index = {f"P{i:05d}": i for i in range(n_targets)}
    (preproc_dir / "target_index.json").write_text(json.dumps(target_index))

    # meta.json
    meta = {
        "n_targets": n_targets,
        "n_standard_types": 3,
        "fp_size": FP_SIZE,
        "fp_type": "binary",
    }
    (preproc_dir / "meta.json").write_text(json.dumps(meta))

    # mol_properties.npz with mean/std for normalisation
    from bind_pred_baseline.preprocess_utils import FEATURE_NAMES as PROP_FEATURE_NAMES
    n_features = len(PROP_FEATURE_NAMES)
    mean = np.zeros(n_features, dtype=np.float32)
    std = np.ones(n_features, dtype=np.float32)
    np.savez(
        preproc_dir / "mol_properties.npz",
        feature_names=np.array(PROP_FEATURE_NAMES),
        mean=mean,
        std=std,
    )

    return preproc_dir


def _make_input_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write an input CSV with ligand_name, uniprot_id, smiles columns."""
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# STANDARD_TYPE_INDEX in constants
# ---------------------------------------------------------------------------


class TestStandardTypeIndexInConstants:
    def test_standard_type_index_is_importable(self) -> None:
        from bind_pred_baseline.constants import STANDARD_TYPE_INDEX
        assert isinstance(STANDARD_TYPE_INDEX, dict)

    def test_standard_type_index_has_expected_keys(self) -> None:
        from bind_pred_baseline.constants import STANDARD_TYPE_INDEX
        assert set(STANDARD_TYPE_INDEX.keys()) == {"IC50", "Ki", "Kd"}

    def test_standard_type_index_values_are_0_1_2(self) -> None:
        from bind_pred_baseline.constants import STANDARD_TYPE_INDEX
        assert set(STANDARD_TYPE_INDEX.values()) == {0, 1, 2}
        assert STANDARD_TYPE_INDEX["IC50"] == 0
        assert STANDARD_TYPE_INDEX["Ki"] == 1
        assert STANDARD_TYPE_INDEX["Kd"] == 2


# ---------------------------------------------------------------------------
# preprocess_data_for_prediction
# ---------------------------------------------------------------------------


class TestPreprocessDataForPrediction:
    """Tests for preprocess_data_for_prediction() in preprocess.py."""

    def _make_valid_rows(self, n_targets: int = 2) -> list[dict]:
        return [
            {"ligand_name": "mol_A", "uniprot_id": "P00000", "smiles": "c1ccccc1"},
            {"ligand_name": "mol_B", "uniprot_id": "P00001", "smiles": "CC(=O)O"},
        ]

    def test_returns_tuple_of_seven(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        result = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert len(result) == 7

    def test_fp_matrix_dtype_float32(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        fp_mat, *_ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert fp_mat.dtype == np.float32

    def test_fp_matrix_shape(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        fp_mat, props_mat, fp_indices, target_indices, std_type_indices, names, uniprot_ids = \
            preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        n_unique = fp_mat.shape[0]
        assert fp_mat.shape == (n_unique, FP_SIZE)

    def test_props_matrix_dtype_float32(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        _, props_mat, *_ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert props_mat.dtype == np.float32

    def test_props_matrix_has_mol_prop_features_columns(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        _, props_mat, *_ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert props_mat.shape[1] == N_MOL_PROP_FEATURES

    def test_fp_indices_dtype_int64(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        _, _, fp_indices, *_ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert fp_indices.dtype == np.int64

    def test_target_indices_dtype_int64(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        _, _, _, target_indices, *_ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert target_indices.dtype == np.int64

    def test_std_type_indices_all_same_value(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        _, _, _, _, std_type_indices, _, _ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert np.all(std_type_indices == STANDARD_TYPE_INDEX["IC50"])

    def test_std_type_indices_ki(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        _, _, _, _, std_type_indices, _, _ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="Ki")
        assert np.all(std_type_indices == STANDARD_TYPE_INDEX["Ki"])

    def test_names_list_of_strings(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        *_, names, uniprot_ids = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_uniprot_ids_list_of_strings(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        *_, names, uniprot_ids = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert isinstance(uniprot_ids, list)
        assert all(isinstance(u, str) for u in uniprot_ids)

    def test_names_and_uniprot_ids_same_length_as_indices(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        _, _, fp_indices, target_indices, std_type_indices, names, uniprot_ids = \
            preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        n = len(fp_indices)
        assert len(target_indices) == n
        assert len(std_type_indices) == n
        assert len(names) == n
        assert len(uniprot_ids) == n

    def test_fp_indices_in_bounds(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        fp_mat, _, fp_indices, *_ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert (fp_indices >= 0).all()
        assert (fp_indices < len(fp_mat)).all()

    def test_target_indices_in_bounds(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        n_targets = 5
        csv = _make_input_csv(tmp_path, self._make_valid_rows(n_targets=n_targets))
        _, _, _, target_indices, *_ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert (target_indices >= 0).all()
        assert (target_indices < n_targets).all()

    def test_unknown_uniprot_id_raises_key_error(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        rows = [{"ligand_name": "mol_A", "uniprot_id": "Q99999", "smiles": "c1ccccc1"}]
        csv = _make_input_csv(tmp_path, rows)
        with pytest.raises(KeyError):
            preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")

    def test_oov_mapping_resolves_unknown_target(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path, n_targets=5)
        # Write oov_target_mapping.json that maps unknown → known target
        oov_mapping = {"Q99999": "P00000"}
        (preproc_dir / "oov_target_mapping.json").write_text(json.dumps(oov_mapping))
        rows = [{"ligand_name": "mol_A", "uniprot_id": "Q99999", "smiles": "c1ccccc1"}]
        csv = _make_input_csv(tmp_path, rows)
        # Should not raise
        result = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert len(result) == 7

    def test_no_oov_mapping_file_is_ok(self, tmp_path: Path) -> None:
        """Missing oov_target_mapping.json should be handled gracefully."""
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path)
        # Confirm no oov file
        assert not (preproc_dir / "oov_target_mapping.json").exists()
        csv = _make_input_csv(tmp_path, self._make_valid_rows())
        # Should not raise
        result = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        assert len(result) == 7

    def test_duplicate_molecules_deduped_in_fp_matrix(self, tmp_path: Path) -> None:
        """Same molecule appearing twice should result in shared fp_matrix row."""
        from bind_pred_baseline.preprocess_pred_data import preprocess_data_for_prediction
        preproc_dir = _make_preproc_dir(tmp_path, n_targets=5)
        rows = [
            {"ligand_name": "mol_A", "uniprot_id": "P00000", "smiles": "c1ccccc1"},
            {"ligand_name": "mol_A", "uniprot_id": "P00001", "smiles": "c1ccccc1"},
        ]
        csv = _make_input_csv(tmp_path, rows)
        fp_mat, _, fp_indices, *_ = preprocess_data_for_prediction(csv, preproc_dir, standard_type="IC50")
        # Both rows have same fp_index pointing to same row
        assert fp_indices[0] == fp_indices[1]


# ---------------------------------------------------------------------------
# PredictDataset
# ---------------------------------------------------------------------------


def _make_predict_dataset(n_rows: int = 8, n_unique: int = 5) -> PredictDataset:
    fps_matrix = torch.rand(n_unique, FP_SIZE)
    props_matrix = torch.randn(n_unique, N_MOL_PROP_FEATURES)
    fp_indices = np.array([i % n_unique for i in range(n_rows)], dtype=np.int64)
    target_indices = np.zeros(n_rows, dtype=np.int64)
    std_type_indices = np.zeros(n_rows, dtype=np.int64)
    names = [f"mol_{i}" for i in range(n_rows)]
    uniprot_ids = [f"P{i:05d}" for i in range(n_rows)]
    return PredictDataset(
        fps_matrix=fps_matrix,
        props_matrix=props_matrix,
        fp_indices=fp_indices,
        target_indices=target_indices,
        standard_type_indices=std_type_indices,
        names=names,
        uniprot_ids=uniprot_ids,
    )


class TestPredictDataset:
    def test_len_equals_number_of_rows(self) -> None:
        ds = _make_predict_dataset(n_rows=8)
        assert len(ds) == 8

    def test_getitem_returns_tuple_of_six(self) -> None:
        ds = _make_predict_dataset(n_rows=4)
        item = ds[0]
        assert len(item) == 6

    def test_getitem_fp_is_tensor(self) -> None:
        ds = _make_predict_dataset()
        fp, *_ = ds[0]
        assert isinstance(fp, torch.Tensor)

    def test_getitem_fp_has_correct_size(self) -> None:
        ds = _make_predict_dataset()
        fp, *_ = ds[0]
        assert fp.shape == (FP_SIZE,)

    def test_getitem_mol_props_is_tensor(self) -> None:
        ds = _make_predict_dataset()
        _, mol_props, *_ = ds[0]
        assert isinstance(mol_props, torch.Tensor)

    def test_getitem_mol_props_has_correct_size(self) -> None:
        ds = _make_predict_dataset()
        _, mol_props, *_ = ds[0]
        assert mol_props.shape == (N_MOL_PROP_FEATURES,)

    def test_getitem_target_idx_is_tensor(self) -> None:
        ds = _make_predict_dataset()
        _, _, target_idx, *_ = ds[0]
        assert isinstance(target_idx, torch.Tensor)

    def test_getitem_std_type_idx_is_tensor(self) -> None:
        ds = _make_predict_dataset()
        _, _, _, std_type_idx, *_ = ds[0]
        assert isinstance(std_type_idx, torch.Tensor)

    def test_getitem_name_is_string(self) -> None:
        ds = _make_predict_dataset()
        *_, name, uniprot_id = ds[0]
        assert isinstance(name, str)

    def test_getitem_uniprot_id_is_string(self) -> None:
        ds = _make_predict_dataset()
        *_, name, uniprot_id = ds[0]
        assert isinstance(uniprot_id, str)

    def test_getitem_name_value_correct(self) -> None:
        ds = _make_predict_dataset(n_rows=4)
        *_, name, _ = ds[2]
        assert name == "mol_2"

    def test_getitem_uniprot_id_value_correct(self) -> None:
        ds = _make_predict_dataset(n_rows=4)
        *_, _, uniprot_id = ds[1]
        assert uniprot_id == "P00001"

    def test_fp_indices_used_for_lookup(self) -> None:
        # Two rows with same fp_index should get identical fingerprints
        fps_matrix = torch.rand(3, FP_SIZE)
        props_matrix = torch.randn(3, N_MOL_PROP_FEATURES)
        fp_indices = np.array([0, 0], dtype=np.int64)
        ds = PredictDataset(
            fps_matrix=fps_matrix,
            props_matrix=props_matrix,
            fp_indices=fp_indices,
            target_indices=np.zeros(2, dtype=np.int64),
            standard_type_indices=np.zeros(2, dtype=np.int64),
            names=["a", "b"],
            uniprot_ids=["P0", "P1"],
        )
        fp0, *_ = ds[0]
        fp1, *_ = ds[1]
        assert torch.allclose(fp0, fp1)


# ---------------------------------------------------------------------------
# AffinityModel.predict_step
# ---------------------------------------------------------------------------


def _make_model_for_predict(n_targets: int = 10) -> AffinityModel:
    return AffinityModel(
        n_targets=n_targets,
        n_standard_types=3,
        hidden_dim=64,
        target_embed_dim=16,
    )


def _make_predict_batch(batch_size: int = 4, n_targets: int = 10):
    fps = torch.rand(batch_size, FP_SIZE)
    mol_props = torch.randn(batch_size, N_MOL_PROP_FEATURES)
    target_idx = torch.randint(0, n_targets, (batch_size,))
    std_type_idx = torch.zeros(batch_size, dtype=torch.long)
    names = [f"mol_{i}" for i in range(batch_size)]
    uniprot_ids = [f"P{i:05d}" for i in range(batch_size)]
    return fps, mol_props, target_idx, std_type_idx, names, uniprot_ids


class TestAffinityModelPredictStep:
    def test_predict_step_returns_dict(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch()
        result = model.predict_step(batch, batch_idx=0)
        assert isinstance(result, dict)

    def test_predict_step_has_name_key(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch()
        result = model.predict_step(batch, batch_idx=0)
        assert "ligand_name" in result

    def test_predict_step_has_uniprot_id_key(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch()
        result = model.predict_step(batch, batch_idx=0)
        assert "uniprot_id" in result

    def test_predict_step_has_pred_pchembl_key(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch()
        result = model.predict_step(batch, batch_idx=0)
        assert "pred_pchembl" in result

    def test_predict_step_pred_pchembl_shape(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch(batch_size=4)
        result = model.predict_step(batch, batch_idx=0)
        assert result["pred_pchembl"].shape == (4,)

    def test_predict_step_names_is_list(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch(batch_size=3)
        result = model.predict_step(batch, batch_idx=0)
        assert isinstance(result["ligand_name"], list)
        assert len(result["ligand_name"]) == 3

    def test_predict_step_uniprot_ids_is_list(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch(batch_size=3)
        result = model.predict_step(batch, batch_idx=0)
        assert isinstance(result["uniprot_id"], list)
        assert len(result["uniprot_id"]) == 3

    def test_predict_step_pred_is_cpu_tensor(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch()
        result = model.predict_step(batch, batch_idx=0)
        assert result["pred_pchembl"].device == torch.device("cpu")

    def test_predict_step_names_match_batch(self) -> None:
        model = _make_model_for_predict()
        batch = _make_predict_batch(batch_size=3)
        result = model.predict_step(batch, batch_idx=0)
        assert result["ligand_name"] == ["mol_0", "mol_1", "mol_2"]


# ---------------------------------------------------------------------------
# PredictWriterCallback
# ---------------------------------------------------------------------------


class TestPredictWriterCallback:
    def test_instantiation(self, tmp_path: Path) -> None:
        output_csv = str(tmp_path / "predictions.csv")
        cb = PredictWriterCallback(output_csv=output_csv)
        assert cb is not None

    def test_write_on_epoch_end_creates_csv(self, tmp_path: Path) -> None:
        output_csv = str(tmp_path / "predictions.csv")
        cb = PredictWriterCallback(output_csv=output_csv)

        # predictions: list[dict] — one dict per batch, as returned by predict_step
        batch1 = {
            "ligand_name": ["mol_A", "mol_B"],
            "uniprot_id": ["P00000", "P00001"],
            "pred_pchembl": torch.tensor([6.5, 7.2]),
        }
        batch2 = {
            "ligand_name": ["mol_C"],
            "uniprot_id": ["P00002"],
            "pred_pchembl": torch.tensor([5.8]),
        }
        predictions = [batch1, batch2]

        trainer = MagicMock()
        pl_module = MagicMock()
        batch_indices = None

        cb.write_on_epoch_end(trainer, pl_module, predictions, batch_indices)

        assert Path(output_csv).exists()

    def test_write_on_epoch_end_csv_columns(self, tmp_path: Path) -> None:
        output_csv = str(tmp_path / "predictions.csv")
        cb = PredictWriterCallback(output_csv=output_csv)

        predictions = [{
            "ligand_name": ["mol_A"],
            "uniprot_id": ["P00000"],
            "pred_pchembl": torch.tensor([6.5]),
        }]

        cb.write_on_epoch_end(MagicMock(), MagicMock(), predictions, None)

        df = pd.read_csv(output_csv)
        assert "ligand_name" in df.columns
        assert "uniprot_id" in df.columns
        assert "pred_pchembl" in df.columns

    def test_write_on_epoch_end_csv_values(self, tmp_path: Path) -> None:
        output_csv = str(tmp_path / "predictions.csv")
        cb = PredictWriterCallback(output_csv=output_csv)

        predictions = [{
            "ligand_name": ["mol_A", "mol_B"],
            "uniprot_id": ["P00000", "P00001"],
            "pred_pchembl": torch.tensor([6.5, 7.2]),
        }]

        cb.write_on_epoch_end(MagicMock(), MagicMock(), predictions, None)

        df = pd.read_csv(output_csv)
        assert len(df) == 2
        assert list(df["ligand_name"]) == ["mol_A", "mol_B"]
        assert list(df["uniprot_id"]) == ["P00000", "P00001"]
        assert pytest.approx(list(df["pred_pchembl"]), abs=1e-4) == [6.5, 7.2]

    def test_write_on_epoch_end_flattens_multiple_batches(self, tmp_path: Path) -> None:
        output_csv = str(tmp_path / "predictions.csv")
        cb = PredictWriterCallback(output_csv=output_csv)

        predictions = [
            {"ligand_name": ["mol_A"], "uniprot_id": ["P00000"], "pred_pchembl": torch.tensor([6.5])},
            {"ligand_name": ["mol_B"], "uniprot_id": ["P00001"], "pred_pchembl": torch.tensor([7.2])},
            {"ligand_name": ["mol_C"], "uniprot_id": ["P00002"], "pred_pchembl": torch.tensor([5.0])},
        ]

        cb.write_on_epoch_end(MagicMock(), MagicMock(), predictions, None)

        df = pd.read_csv(output_csv)
        assert len(df) == 3

    def test_write_on_batch_end_does_not_raise(self, tmp_path: Path) -> None:
        cb = PredictWriterCallback(output_csv=str(tmp_path / "predictions.csv"))
        # write_on_batch_end is a required abstract method — should be a no-op
        cb.write_on_batch_end(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# model_utils module (renamed from evaluate.py)
# ---------------------------------------------------------------------------


class TestModelUtilsModule:
    def test_pearson_r_importable_from_model_utils(self) -> None:
        from bind_pred_baseline.model_utils import pearson_r_per_assay
        assert callable(pearson_r_per_assay)

    def test_metrics_plot_callback_importable_from_model_utils(self) -> None:
        from bind_pred_baseline.model_utils import MetricsPlotCallback
        assert MetricsPlotCallback is not None
