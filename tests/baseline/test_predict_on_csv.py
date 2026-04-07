"""Tests for bind_pred_baseline.predict_on_csv."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bind_pred_baseline.constants import FP_SIZE
from bind_pred_baseline.model import AffinityModel


# ---------------------------------------------------------------------------
# Helpers (shared with test_evaluate.py)
# ---------------------------------------------------------------------------

_SMILES = ["c1ccccc1", "CC(=O)O", "CCO", "c1ccncc1", "CCCC"]


def _make_preproc_dir(tmp_path: Path, n_targets: int = 5) -> Path:
    from bind_pred_baseline.preprocess_utils import FEATURE_NAMES as PROP_FEATURE_NAMES

    d = tmp_path / "preproc"
    d.mkdir(parents=True, exist_ok=True)
    target_index = {f"P{i:05d}": i for i in range(n_targets)}
    (d / "target_index.json").write_text(json.dumps(target_index))
    meta = {
        "n_targets": n_targets,
        "n_standard_types": 3,
        "fp_size": FP_SIZE,
        "fp_type": "binary",
    }
    (d / "meta.json").write_text(json.dumps(meta))
    n_features = len(PROP_FEATURE_NAMES)
    np.savez(
        d / "mol_properties.npz",
        feature_names=np.array(PROP_FEATURE_NAMES),
        mean=np.zeros(n_features, dtype=np.float32),
        std=np.ones(n_features, dtype=np.float32),
    )
    return d


def _save_tiny_checkpoint(path: Path, n_targets: int = 5) -> None:
    import lightning

    model = AffinityModel(
        n_targets=n_targets, n_standard_types=3, hidden_dim=32, target_embed_dim=8
    )
    torch.save(
        {
            "epoch": 0,
            "global_step": 0,
            "pytorch-lightning_version": lightning.__version__,
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "callbacks": {},
            "optimizer_states": [],
            "lr_schedulers": [],
        },
        str(path),
    )


def _make_input_csv(
    tmp_path: Path, rows: list[dict], filename: str = "input.csv"
) -> Path:
    csv_path = tmp_path / filename
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# load_csv_as_standard_df
# ---------------------------------------------------------------------------


class TestLoadCsvAsStandardDf:
    def _make_rows(self, n: int = 3) -> list[dict]:
        return [
            {
                "ligand_name": f"mol_{i}",
                "uniprot_id": f"P{i:05d}",
                "smiles": _SMILES[i % len(_SMILES)],
            }
            for i in range(n)
        ]

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        csv = _make_input_csv(tmp_path, self._make_rows())
        df = load_csv_as_standard_df(csv)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        csv = _make_input_csv(tmp_path, self._make_rows())
        df = load_csv_as_standard_df(csv)
        for col in [
            "ligand_name",
            "smiles",
            "uniprot_id",
            "standard_type",
            "assay_id",
            "split",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_default_standard_type_ic50(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        csv = _make_input_csv(tmp_path, self._make_rows())
        df = load_csv_as_standard_df(csv, default_standard_type="IC50")
        assert (df["standard_type"] == "IC50").all()

    def test_default_standard_type_ki(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        csv = _make_input_csv(tmp_path, self._make_rows())
        df = load_csv_as_standard_df(csv, default_standard_type="Ki")
        assert (df["standard_type"] == "Ki").all()

    def test_standard_type_column_in_csv_takes_precedence(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        rows = [
            {
                "ligand_name": "mol_0",
                "uniprot_id": "P00000",
                "smiles": "c1ccccc1",
                "standard_type": "Ki",
            },
            {
                "ligand_name": "mol_1",
                "uniprot_id": "P00001",
                "smiles": "CCO",
                "standard_type": "Kd",
            },
        ]
        csv = _make_input_csv(tmp_path, rows)
        df = load_csv_as_standard_df(csv, default_standard_type="IC50")
        assert list(df["standard_type"]) == ["Ki", "Kd"]

    def test_assay_id_equals_uniprot_id(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        csv = _make_input_csv(tmp_path, self._make_rows())
        df = load_csv_as_standard_df(csv)
        assert (df["assay_id"] == df["uniprot_id"]).all()

    def test_split_is_predict(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        csv = _make_input_csv(tmp_path, self._make_rows())
        df = load_csv_as_standard_df(csv)
        assert (df["split"] == "predict").all()

    def test_accepts_canonical_smiles_column(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        rows = [
            {
                "ligand_name": "mol_0",
                "uniprot_id": "P00000",
                "canonical_smiles": "c1ccccc1",
            },
        ]
        csv = _make_input_csv(tmp_path, rows)
        df = load_csv_as_standard_df(csv)
        assert "smiles" in df.columns
        assert "canonical_smiles" not in df.columns

    def test_pchembl_value_absent_when_not_in_csv(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        csv = _make_input_csv(tmp_path, self._make_rows())
        df = load_csv_as_standard_df(csv)
        assert "pchembl_value" not in df.columns

    def test_pchembl_value_preserved_when_in_csv(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import load_csv_as_standard_df

        rows = [
            {
                "ligand_name": "mol_0",
                "uniprot_id": "P00000",
                "smiles": "c1ccccc1",
                "pchembl_value": 7.5,
            },
            {
                "ligand_name": "mol_1",
                "uniprot_id": "P00001",
                "smiles": "CCO",
                "pchembl_value": 6.2,
            },
        ]
        csv = _make_input_csv(tmp_path, rows)
        df = load_csv_as_standard_df(csv)
        assert "pchembl_value" in df.columns
        assert list(df["pchembl_value"]) == [7.5, 6.2]


# ---------------------------------------------------------------------------
# predict_on_csv integration
# ---------------------------------------------------------------------------


OUTPUT_COLUMNS = [
    "assay_id",
    "ligand_name",
    "uniprot_id",
    "standard_type",
    "split",
    "pchembl_value",
    "pred_pchembl",
]


class TestPredictOnCsvIntegration:
    def _setup(self, tmp_path: Path, n_targets: int = 3):
        data_dir = _make_preproc_dir(tmp_path, n_targets=n_targets)
        ckpt_path = tmp_path / "model.ckpt"
        _save_tiny_checkpoint(ckpt_path, n_targets=n_targets)
        rows = [
            {
                "ligand_name": f"mol_{i}",
                "uniprot_id": f"P{i % n_targets:05d}",
                "smiles": _SMILES[i % len(_SMILES)],
            }
            for i in range(5)
        ]
        csv_path = _make_input_csv(tmp_path, rows)
        return data_dir, ckpt_path, csv_path

    def test_creates_output_csv(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv)
        assert out_csv.exists()

    def test_output_has_unified_columns(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv)
        df = pd.read_csv(out_csv)
        for col in OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_pred_pchembl_numeric_and_not_null(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv)
        df = pd.read_csv(out_csv)
        assert pd.api.types.is_float_dtype(df["pred_pchembl"])
        assert df["pred_pchembl"].notna().all()

    def test_split_column_is_predict(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv)
        df = pd.read_csv(out_csv)
        assert (df["split"] == "predict").all()

    def test_pchembl_value_is_nan_when_no_labels(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv)
        df = pd.read_csv(out_csv)
        assert df["pchembl_value"].isna().all()

    def test_assay_id_equals_uniprot_id(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv)
        df = pd.read_csv(out_csv)
        assert (df["assay_id"] == df["uniprot_id"]).all()

    def test_no_metrics_printed_when_no_labels(self, tmp_path: Path, capsys) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv)
        out = capsys.readouterr().out
        assert "Pearson" not in out

    def test_standard_type_default_applied(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv, standard_type="Ki")
        df = pd.read_csv(out_csv)
        assert (df["standard_type"] == "Ki").all()

    def test_output_row_count_matches_input(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        data_dir, ckpt, csv = self._setup(tmp_path)
        n_input = len(pd.read_csv(csv))
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt, data_dir, csv, out_csv)
        df = pd.read_csv(out_csv)
        assert len(df) == n_input

    def test_metrics_printed_when_pchembl_value_in_input(
        self, tmp_path: Path, capsys
    ) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        n_targets = 3
        data_dir = _make_preproc_dir(tmp_path, n_targets=n_targets)
        ckpt_path = tmp_path / "model.ckpt"
        _save_tiny_checkpoint(ckpt_path, n_targets=n_targets)
        rows = [
            {
                "ligand_name": f"mol_{i}",
                "uniprot_id": f"P{i % n_targets:05d}",
                "smiles": _SMILES[i % len(_SMILES)],
                "pchembl_value": 6.0 + i * 0.1,
            }
            for i in range(5)
        ]
        csv = _make_input_csv(tmp_path, rows)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt_path, data_dir, csv, out_csv)
        out = capsys.readouterr().out
        assert "Pearson" in out

    def test_pchembl_value_preserved_in_output_when_in_input(
        self, tmp_path: Path
    ) -> None:
        from bind_pred_baseline.predict_on_csv import predict_on_csv

        n_targets = 3
        data_dir = _make_preproc_dir(tmp_path, n_targets=n_targets)
        ckpt_path = tmp_path / "model.ckpt"
        _save_tiny_checkpoint(ckpt_path, n_targets=n_targets)
        rows = [
            {
                "ligand_name": f"mol_{i}",
                "uniprot_id": f"P{i % n_targets:05d}",
                "smiles": _SMILES[i % len(_SMILES)],
                "pchembl_value": 7.0,
            }
            for i in range(5)
        ]
        csv = _make_input_csv(tmp_path, rows)
        out_csv = tmp_path / "predictions.csv"
        predict_on_csv(ckpt_path, data_dir, csv, out_csv)
        df = pd.read_csv(out_csv)
        assert df["pchembl_value"].notna().all()
