"""Tests for bind_pred_baseline.predict_on_benchmark and bind_pred_baseline.preprocess_pred_data.

Covers:
- preprocess_for_inference (shared preprocessing, standard DataFrame in)
- load_activities_as_standard_df (thin benchmark entry-point)
- run_inference (shared inference loop)
- _print_metrics (metrics reporting with optional bootstrap)
- evaluate_splits (end-to-end integration)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bind_pred_baseline.constants import FP_SIZE, N_MOL_PROP_FEATURES, STANDARD_TYPE_INDEX
from bind_pred_baseline.model import AffinityModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SMILES = ["c1ccccc1", "CC(=O)O", "CCO", "c1ccncc1", "CCCC", "CN(C)C", "c1ccc(N)cc1"]


def _make_preproc_dir(tmp_path: Path, n_targets: int = 5) -> Path:
    """Minimal data_dir compatible with training preprocessing artifacts."""
    from bind_pred_baseline.preprocess_utils import FEATURE_NAMES as PROP_FEATURE_NAMES

    d = tmp_path / "preproc"
    d.mkdir(parents=True, exist_ok=True)

    target_index = {f"P{i:05d}": i for i in range(n_targets)}
    (d / "target_index.json").write_text(json.dumps(target_index))

    meta = {"n_targets": n_targets, "n_standard_types": 3, "fp_size": FP_SIZE, "fp_type": "binary"}
    (d / "meta.json").write_text(json.dumps(meta))

    n_features = len(PROP_FEATURE_NAMES)
    np.savez(
        d / "mol_properties.npz",
        feature_names=np.array(PROP_FEATURE_NAMES),
        mean=np.zeros(n_features, dtype=np.float32),
        std=np.ones(n_features, dtype=np.float32),
    )
    return d


def _make_standard_df(
    n_rows: int = 10,
    split: str = "test",
    n_targets: int = 3,
    mixed_standard_types: bool = False,
    include_labels: bool = True,
) -> pd.DataFrame:
    """Standard input DataFrame (ligand_name, smiles, uniprot_id, standard_type, ...)."""
    std_types = ["IC50", "Ki", "Kd"] if mixed_standard_types else ["IC50"] * n_rows
    rows = []
    for i in range(n_rows):
        row = {
            "assay_id": f"ASSAY{100 + i % 3}",
            "ligand_name": f"MOL{200 + i}",
            "uniprot_id": f"P{(i % n_targets):05d}",
            "standard_type": std_types[i % len(std_types)],
            "split": split,
            "smiles": _SMILES[i % len(_SMILES)],
        }
        if include_labels:
            row["pchembl_value"] = 6.0 + i * 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def _make_activities_parquet(path: Path, splits: list[str] | None = None, n_targets: int = 3) -> None:
    """Write a minimal activities.parquet (without uniprot_id — uses target_chembl_id).

    Generates 20 rows per split across 2 assays (10 rows each) so that
    MIN_ASSAY_SIZE=10 is met and Pearson r metrics are computable.
    """
    if splits is None:
        splits = ["test"]
    rows = []
    for split in splits:
        for i in range(20):
            rows.append({
                "target_chembl_id": f"CHEMBL{i % n_targets}",
                "assay_chembl_id": f"CHEMBL{100 + i % 2}",
                "ligand_chembl_id": f"CHEMBL{200 + len(rows)}",
                "standard_type": ["IC50", "Ki"][i % 2],
                "pchembl_relation": "=",
                "pchembl_value_filled": 6.0 + i * 0.1,
                "split": split,
                "canonical_smiles": _SMILES[i % len(_SMILES)],
            })
    pd.DataFrame(rows).to_parquet(path, index=False)


def _make_targets_parquet(path: Path, n_targets: int = 3) -> None:
    rows = [
        {"target_chembl_id": f"CHEMBL{i}", "uniprot_id": f"P{i:05d}", "gene_name": f"GENE{i}"}
        for i in range(n_targets)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _save_tiny_checkpoint(path: Path, n_targets: int = 5) -> None:
    """Save a minimal Lightning-compatible checkpoint for AffinityModel."""
    import lightning
    model = AffinityModel(
        n_targets=n_targets,
        n_standard_types=3,
        hidden_dim=32,
        target_embed_dim=8,
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


# ---------------------------------------------------------------------------
# preprocess_for_inference
# ---------------------------------------------------------------------------


class TestPreprocessForInference:
    def test_returns_six_tuple(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        data_dir = _make_preproc_dir(tmp_path)
        result = preprocess_for_inference(_make_standard_df(), data_dir)
        assert len(result) == 6

    def test_fp_matrix_dtype_float32(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        fp_matrix, *_ = preprocess_for_inference(_make_standard_df(), _make_preproc_dir(tmp_path))
        assert fp_matrix.dtype == np.float32

    def test_fp_matrix_second_dim_is_fp_size(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        fp_matrix, *_ = preprocess_for_inference(_make_standard_df(), _make_preproc_dir(tmp_path))
        assert fp_matrix.shape[1] == FP_SIZE

    def test_props_matrix_dtype_float32(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        _, props_matrix, *_ = preprocess_for_inference(_make_standard_df(), _make_preproc_dir(tmp_path))
        assert props_matrix.dtype == np.float32

    def test_props_matrix_second_dim_is_n_mol_prop_features(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        _, props_matrix, *_ = preprocess_for_inference(_make_standard_df(), _make_preproc_dir(tmp_path))
        assert props_matrix.shape[1] == N_MOL_PROP_FEATURES

    def test_fp_indices_dtype_int64(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        _, _, fp_indices, *_ = preprocess_for_inference(_make_standard_df(), _make_preproc_dir(tmp_path))
        assert fp_indices.dtype == np.int64

    def test_target_indices_dtype_int64(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        _, _, _, target_indices, *_ = preprocess_for_inference(_make_standard_df(), _make_preproc_dir(tmp_path))
        assert target_indices.dtype == np.int64

    def test_std_type_indices_dtype_int64(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        _, _, _, _, std_type_indices, _ = preprocess_for_inference(_make_standard_df(), _make_preproc_dir(tmp_path))
        assert std_type_indices.dtype == np.int64

    def test_arrays_and_df_filtered_same_length(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        df = _make_standard_df(n_rows=8)
        _, _, fp_indices, target_indices, std_type_indices, df_filtered = \
            preprocess_for_inference(df, _make_preproc_dir(tmp_path))
        n = len(df_filtered)
        assert len(fp_indices) == n
        assert len(target_indices) == n
        assert len(std_type_indices) == n

    def test_fp_indices_in_bounds(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        fp_matrix, _, fp_indices, *_ = preprocess_for_inference(_make_standard_df(), _make_preproc_dir(tmp_path))
        assert (fp_indices >= 0).all()
        assert (fp_indices < len(fp_matrix)).all()

    def test_target_indices_in_bounds(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        n_targets = 5
        _, _, _, target_indices, *_ = preprocess_for_inference(
            _make_standard_df(n_targets=n_targets), _make_preproc_dir(tmp_path, n_targets=n_targets)
        )
        assert (target_indices >= 0).all()
        assert (target_indices < n_targets).all()

    def test_std_type_indices_reflect_per_row_standard_type(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        df = _make_standard_df(n_rows=9, mixed_standard_types=True)
        _, _, _, _, std_type_indices, _ = preprocess_for_inference(df, _make_preproc_dir(tmp_path))
        assert len(set(std_type_indices.tolist())) > 1

    def test_std_type_index_values_match_constants(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        df = _make_standard_df(n_rows=9, mixed_standard_types=True)
        _, _, _, _, std_type_indices, df_filtered = preprocess_for_inference(df, _make_preproc_dir(tmp_path))
        expected = [STANDARD_TYPE_INDEX[st] for st in df_filtered["standard_type"]]
        assert std_type_indices.tolist() == expected

    def test_df_filtered_preserves_input_columns(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        df = _make_standard_df()
        _, _, _, _, _, df_filtered = preprocess_for_inference(df, _make_preproc_dir(tmp_path))
        for col in df.columns:
            assert col in df_filtered.columns

    def test_smiles_parse_failure_drops_rows(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        df = _make_standard_df(n_rows=4).copy()
        df.loc[0, "smiles"] = "not_a_valid_smiles!!!!"
        _, _, _, _, _, df_filtered = preprocess_for_inference(df, _make_preproc_dir(tmp_path))
        bad_cpd = df.loc[0, "ligand_name"]
        assert bad_cpd not in df_filtered["ligand_name"].values

    def test_unknown_target_raises_key_error(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        df = _make_standard_df(n_rows=3, n_targets=3).copy()
        df.loc[0, "uniprot_id"] = "QUNKNWN"
        with pytest.raises(KeyError):
            preprocess_for_inference(df, _make_preproc_dir(tmp_path, n_targets=3))

    def test_key_error_lists_all_unresolvable_targets(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        df = _make_standard_df(n_rows=4, n_targets=3).copy()
        df["uniprot_id"] = ["QAAA", "QBBB", "P00000", "QAAA"]
        with pytest.raises(KeyError) as exc_info:
            preprocess_for_inference(df, _make_preproc_dir(tmp_path, n_targets=3))
        msg = str(exc_info.value)
        assert "QAAA" in msg
        assert "QBBB" in msg

    def test_oov_mapping_resolves_unknown_target(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        data_dir = _make_preproc_dir(tmp_path, n_targets=5)
        (data_dir / "oov_target_mapping.json").write_text(json.dumps({"QUNKNWN": "P00000"}))
        df = _make_standard_df(n_rows=3).copy()
        df["uniprot_id"] = "QUNKNWN"
        result = preprocess_for_inference(df, data_dir)
        assert len(result) == 6

    def test_missing_oov_file_is_ok(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        data_dir = _make_preproc_dir(tmp_path)
        assert not (data_dir / "oov_target_mapping.json").exists()
        result = preprocess_for_inference(_make_standard_df(), data_dir)
        assert len(result) == 6

    def test_accepts_canonical_smiles_column(self, tmp_path: Path) -> None:
        """DataFrame with canonical_smiles instead of smiles should still work."""
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        df = _make_standard_df().rename(columns={"smiles": "canonical_smiles"})
        result = preprocess_for_inference(df, _make_preproc_dir(tmp_path))
        assert len(result) == 6

    def test_duplicate_ligand_names_share_fp_matrix_row(self, tmp_path: Path) -> None:
        from bind_pred_baseline.preprocess_pred_data import preprocess_for_inference

        data_dir = _make_preproc_dir(tmp_path, n_targets=5)
        df = pd.DataFrame([
            {"ligand_name": "mol_A", "smiles": "c1ccccc1", "uniprot_id": "P00000", "standard_type": "IC50"},
            {"ligand_name": "mol_A", "smiles": "c1ccccc1", "uniprot_id": "P00001", "standard_type": "IC50"},
        ])
        fp_matrix, _, fp_indices, *_ = preprocess_for_inference(df, data_dir)
        assert fp_indices[0] == fp_indices[1]


# ---------------------------------------------------------------------------
# load_activities_as_standard_df
# ---------------------------------------------------------------------------


class TestLoadActivitiesAsStandardDf:
    def _setup(self, tmp_path: Path, n_targets: int = 3, splits: list[str] | None = None):
        acts_path = tmp_path / "activities.parquet"
        tgts_path = tmp_path / "targets.parquet"
        _make_activities_parquet(acts_path, splits=splits or ["test"], n_targets=n_targets)
        _make_targets_parquet(tgts_path, n_targets=n_targets)
        return acts_path, tgts_path

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        assert isinstance(df, pd.DataFrame)

    def test_has_required_standard_columns(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        for col in ["ligand_name", "smiles", "uniprot_id", "standard_type"]:
            assert col in df.columns, f"Missing required column: {col}"

    def test_renames_assay_chembl_id_to_assay_id(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        assert "assay_id" in df.columns
        assert "assay_chembl_id" not in df.columns

    def test_renames_ligand_chembl_id_to_ligand_name(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        assert "ligand_name" in df.columns
        assert "ligand_chembl_id" not in df.columns

    def test_renames_canonical_smiles_to_smiles(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        assert "smiles" in df.columns
        assert "canonical_smiles" not in df.columns

    def test_filters_to_requested_split(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path, splits=["test", "val"])
        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        assert set(df["split"].unique()) == {"test"}

    def test_multiple_splits_included(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path, splits=["test", "val"])
        df = load_activities_as_standard_df(acts, tgts, splits=["test", "val"])
        assert set(df["split"].unique()) == {"test", "val"}

    def test_joins_uniprot_id_from_targets(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        assert df["uniprot_id"].notna().all()
        assert df["uniprot_id"].str.startswith("P").all()

    def test_raises_when_no_rows_for_any_split(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        with pytest.raises(ValueError, match="No rows found"):
            load_activities_as_standard_df(acts, tgts, splits=["nonexistent"])

    def test_drops_null_pchembl_value(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        # Inject null into parquet (source column is still pchembl_value_filled)
        df_acts = pd.read_parquet(acts)
        df_acts.loc[0, "pchembl_value_filled"] = None
        df_acts.to_parquet(acts, index=False)

        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        assert "pchembl_value" in df.columns
        assert df["pchembl_value"].notna().all()

    def test_renames_pchembl_value_filled_to_pchembl_value(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import load_activities_as_standard_df

        acts, tgts = self._setup(tmp_path)
        df = load_activities_as_standard_df(acts, tgts, splits=["test"])
        assert "pchembl_value" in df.columns
        assert "pchembl_value_filled" not in df.columns


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------


class TestRunInference:
    def _make_arrays(self, n_rows: int = 8, n_unique: int = 5, n_targets: int = 3):
        rng = np.random.default_rng(0)
        fp_matrix = rng.random((n_unique, FP_SIZE), dtype=np.float64).astype(np.float32)
        props_matrix = rng.random((n_unique, N_MOL_PROP_FEATURES), dtype=np.float64).astype(np.float32)
        fp_indices = np.array([i % n_unique for i in range(n_rows)], dtype=np.int64)
        target_indices = np.zeros(n_rows, dtype=np.int64)
        std_type_indices = np.zeros(n_rows, dtype=np.int64)
        return fp_matrix, props_matrix, fp_indices, target_indices, std_type_indices

    def _make_model(self, n_targets: int = 3) -> AffinityModel:
        return AffinityModel(n_targets=n_targets, n_standard_types=3, hidden_dim=32, target_embed_dim=8)

    def test_returns_ndarray(self) -> None:
        from bind_pred_baseline.predict_on_benchmark import run_inference

        model = self._make_model()
        arrays = self._make_arrays()
        result = run_inference(model, *arrays)
        assert isinstance(result, np.ndarray)

    def test_returns_float32(self) -> None:
        from bind_pred_baseline.predict_on_benchmark import run_inference

        model = self._make_model()
        arrays = self._make_arrays()
        result = run_inference(model, *arrays)
        assert result.dtype == np.float32

    def test_length_matches_n_rows(self) -> None:
        from bind_pred_baseline.predict_on_benchmark import run_inference

        n_rows = 12
        model = self._make_model()
        arrays = self._make_arrays(n_rows=n_rows)
        result = run_inference(model, *arrays)
        assert len(result) == n_rows

    def test_device_none_does_not_raise(self) -> None:
        from bind_pred_baseline.predict_on_benchmark import run_inference

        model = self._make_model()
        arrays = self._make_arrays()
        result = run_inference(model, *arrays, device=None)
        assert len(result) > 0

    def test_explicit_cpu_device(self) -> None:
        from bind_pred_baseline.predict_on_benchmark import run_inference

        model = self._make_model()
        arrays = self._make_arrays()
        result = run_inference(model, *arrays, device="cpu")
        assert isinstance(result, np.ndarray)

    def test_no_nan_in_output(self) -> None:
        from bind_pred_baseline.predict_on_benchmark import run_inference

        model = self._make_model()
        arrays = self._make_arrays()
        result = run_inference(model, *arrays)
        assert not np.isnan(result).any()


# ---------------------------------------------------------------------------
# _print_metrics
# ---------------------------------------------------------------------------


class TestPrintMetrics:
    def _make_df(self, n_rows: int = 30) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "assay_id": [f"ASSAY{i % 3}" for i in range(n_rows)],
            "standard_type": ["IC50"] * n_rows,
            "split": ["test"] * n_rows,
            "pchembl_value": rng.uniform(4, 9, n_rows).astype(np.float32),
            "pred_pchembl": rng.uniform(4, 9, n_rows).astype(np.float32),
        })

    def test_runs_without_error(self, capsys) -> None:
        from bind_pred_baseline.predict_on_benchmark import _print_metrics

        _print_metrics(self._make_df())
        captured = capsys.readouterr()
        assert "Pearson" in captured.out

    def test_prints_each_split(self, capsys) -> None:
        from bind_pred_baseline.predict_on_benchmark import _print_metrics

        rng = np.random.default_rng(1)
        n = 30
        df = pd.DataFrame({
            "assay_id": [f"ASSAY{i % 3}" for i in range(n)],
            "standard_type": ["IC50"] * n,
            "split": ["test"] * 15 + ["2024_not_novel"] * 15,
            "pchembl_value": rng.uniform(4, 9, n).astype(np.float32),
            "pred_pchembl": rng.uniform(4, 9, n).astype(np.float32),
        })
        _print_metrics(df)
        out = capsys.readouterr().out
        assert "test" in out
        assert "2024_not_novel" in out

    def test_prints_overall_when_multiple_splits(self, capsys) -> None:
        from bind_pred_baseline.predict_on_benchmark import _print_metrics

        rng = np.random.default_rng(2)
        n = 30
        df = pd.DataFrame({
            "assay_id": [f"ASSAY{i % 3}" for i in range(n)],
            "standard_type": ["IC50"] * n,
            "split": ["test"] * 15 + ["2024_not_novel"] * 15,
            "pchembl_value": rng.uniform(4, 9, n).astype(np.float32),
            "pred_pchembl": rng.uniform(4, 9, n).astype(np.float32),
        })
        _print_metrics(df)
        out = capsys.readouterr().out
        assert "overall" in out

    def test_no_overall_line_for_single_split(self, capsys) -> None:
        from bind_pred_baseline.predict_on_benchmark import _print_metrics

        _print_metrics(self._make_df())
        out = capsys.readouterr().out
        assert "overall" not in out

    def test_bootstrap_se_printed_when_n_bootstrap_given(self, capsys) -> None:
        from bind_pred_baseline.predict_on_benchmark import _print_metrics

        _print_metrics(self._make_df(), n_bootstrap=50)
        out = capsys.readouterr().out
        assert "±" in out

    def test_no_se_printed_when_n_bootstrap_is_none(self, capsys) -> None:
        from bind_pred_baseline.predict_on_benchmark import _print_metrics

        _print_metrics(self._make_df(), n_bootstrap=None)
        out = capsys.readouterr().out
        assert "±" not in out


# ---------------------------------------------------------------------------
# evaluate_splits (integration)
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


class TestEvaluateSplits:
    def _setup(self, tmp_path: Path, n_targets: int = 3):
        data_dir = _make_preproc_dir(tmp_path, n_targets=n_targets)
        ckpt_path = tmp_path / "model.ckpt"
        _save_tiny_checkpoint(ckpt_path, n_targets=n_targets)
        acts_path = tmp_path / "activities.parquet"
        _make_activities_parquet(acts_path, splits=["test", "2024_not_novel"], n_targets=n_targets)
        targets_path = tmp_path / "targets.parquet"
        _make_targets_parquet(targets_path, n_targets=n_targets)
        return data_dir, ckpt_path, acts_path, targets_path

    def test_creates_output_csv(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        assert out_csv.exists()

    def test_output_csv_has_unified_columns(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        for col in OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_output_uses_assay_id_not_assay_chembl_id(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert "assay_id" in df.columns
        assert "assay_chembl_id" not in df.columns

    def test_output_uses_ligand_name_not_ligand_chembl_id(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert "ligand_name" in df.columns
        assert "ligand_chembl_id" not in df.columns

    def test_output_rows_match_requested_split(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        activities_df = pd.read_parquet(acts)
        expected_n = len(activities_df[activities_df["split"] == "test"])

        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        result = pd.read_csv(out_csv)
        assert len(result) == expected_n

    def test_multiple_splits_in_output(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test", "2024_not_novel"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert set(df["split"].unique()) == {"test", "2024_not_novel"}

    def test_pred_pchembl_is_numeric_and_not_null(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert pd.api.types.is_float_dtype(df["pred_pchembl"])
        assert df["pred_pchembl"].notna().all()

    def test_raises_when_no_rows_for_any_split(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        with pytest.raises(ValueError, match="No rows found"):
            evaluate_splits(ckpt, data_dir, acts, tgts, splits=["nonexistent_split"], output_csv=out_csv)

    def test_split_column_preserved_in_output(self, tmp_path: Path) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert (df["split"] == "test").all()

    def test_prints_pearson_r(self, tmp_path: Path, capsys) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        out = capsys.readouterr().out
        assert "Pearson" in out

    def test_n_bootstrap_prints_se(self, tmp_path: Path, capsys) -> None:
        from bind_pred_baseline.predict_on_benchmark import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv, n_bootstrap=50)
        out = capsys.readouterr().out
        assert "±" in out
