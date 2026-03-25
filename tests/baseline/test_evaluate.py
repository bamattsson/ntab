"""Tests for bind_pred_baseline.evaluate: preprocess_activities_for_eval,
_print_metrics, and evaluate_splits."""

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


def _make_activities_df(
    n_rows: int = 10,
    split: str = "test",
    n_targets: int = 3,
    mixed_standard_types: bool = False,
) -> pd.DataFrame:
    """Minimal activities-like DataFrame (post-join with targets — has uniprot_id)."""
    std_types = ["IC50", "Ki", "Kd"] if mixed_standard_types else ["IC50"] * n_rows
    rows = []
    for i in range(n_rows):
        rows.append({
            "assay_chembl_id": f"CHEMBL{100 + i % 3}",
            "ligand_chembl_id": f"CHEMBL{200 + i}",
            "uniprot_id": f"P{(i % n_targets):05d}",
            "standard_type": std_types[i % len(std_types)],
            "split": split,
            "pchembl_value_filled": 6.0 + i * 0.1,
            "canonical_smiles": _SMILES[i % len(_SMILES)],
        })
    return pd.DataFrame(rows)


def _make_activities_parquet(path: Path, splits: list[str] | None = None, n_targets: int = 3) -> None:
    """Write a minimal activities.parquet (without uniprot_id — uses target_chembl_id)."""
    if splits is None:
        splits = ["test"]
    rows = []
    for split in splits:
        for i in range(6):
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
# preprocess_activities_for_eval
# ---------------------------------------------------------------------------


class TestPreprocessActivitiesForEval:
    def test_returns_six_tuple(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        df = _make_activities_df()
        result = preprocess_activities_for_eval(df, data_dir)
        assert len(result) == 6

    def test_fp_matrix_dtype_float32(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        fp_matrix, *_ = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert fp_matrix.dtype == np.float32

    def test_fp_matrix_second_dim_is_fp_size(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        fp_matrix, *_ = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert fp_matrix.shape[1] == FP_SIZE

    def test_props_matrix_dtype_float32(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        _, props_matrix, *_ = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert props_matrix.dtype == np.float32

    def test_props_matrix_second_dim_is_n_mol_prop_features(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        _, props_matrix, *_ = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert props_matrix.shape[1] == N_MOL_PROP_FEATURES

    def test_fp_indices_dtype_int64(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        _, _, fp_indices, *_ = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert fp_indices.dtype == np.int64

    def test_target_indices_dtype_int64(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        _, _, _, target_indices, *_ = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert target_indices.dtype == np.int64

    def test_std_type_indices_dtype_int64(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        _, _, _, _, std_type_indices, _ = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert std_type_indices.dtype == np.int64

    def test_arrays_and_df_eval_same_length(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        df = _make_activities_df(n_rows=8)
        _, _, fp_indices, target_indices, std_type_indices, df_eval = \
            preprocess_activities_for_eval(df, data_dir)
        n = len(df_eval)
        assert len(fp_indices) == n
        assert len(target_indices) == n
        assert len(std_type_indices) == n

    def test_fp_indices_in_bounds(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        fp_matrix, _, fp_indices, *_ = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert (fp_indices >= 0).all()
        assert (fp_indices < len(fp_matrix)).all()

    def test_target_indices_in_bounds(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        n_targets = 5
        data_dir = _make_preproc_dir(tmp_path, n_targets=n_targets)
        _, _, _, target_indices, *_ = preprocess_activities_for_eval(
            _make_activities_df(n_targets=n_targets), data_dir
        )
        assert (target_indices >= 0).all()
        assert (target_indices < n_targets).all()

    def test_std_type_indices_reflect_per_row_standard_type(self, tmp_path: Path) -> None:
        """Mixed standard_types should produce different std_type_indices per row."""
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        df = _make_activities_df(n_rows=9, mixed_standard_types=True)
        _, _, _, _, std_type_indices, _ = preprocess_activities_for_eval(df, data_dir)
        assert len(set(std_type_indices.tolist())) > 1

    def test_std_type_index_values_match_constants(self, tmp_path: Path) -> None:
        """Each row's std_type_index should equal STANDARD_TYPE_INDEX[standard_type]."""
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        df = _make_activities_df(n_rows=9, mixed_standard_types=True)
        _, _, _, _, std_type_indices, df_eval = preprocess_activities_for_eval(df, data_dir)
        expected = [STANDARD_TYPE_INDEX[st] for st in df_eval["standard_type"]]
        assert std_type_indices.tolist() == expected

    def test_df_eval_preserves_input_columns(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        df = _make_activities_df()
        _, _, _, _, _, df_eval = preprocess_activities_for_eval(df, data_dir)
        for col in df.columns:
            assert col in df_eval.columns

    def test_smiles_parse_failure_drops_rows(self, tmp_path: Path) -> None:
        """Rows whose SMILES cannot be parsed should be absent from df_eval."""
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        df = _make_activities_df(n_rows=4)
        df = df.copy()
        df.loc[0, "canonical_smiles"] = "not_a_valid_smiles!!!!"
        _, _, _, _, _, df_eval = preprocess_activities_for_eval(df, data_dir)
        bad_cpd = df.loc[0, "ligand_chembl_id"]
        assert bad_cpd not in df_eval["ligand_chembl_id"].values

    def test_unknown_target_raises_key_error(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path, n_targets=3)
        df = _make_activities_df(n_rows=3, n_targets=3)
        df = df.copy()
        df.loc[0, "uniprot_id"] = "QUNKNWN"
        with pytest.raises(KeyError):
            preprocess_activities_for_eval(df, data_dir)

    def test_key_error_lists_all_unresolvable_targets(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path, n_targets=3)
        df = _make_activities_df(n_rows=4, n_targets=3)
        df = df.copy()
        df["uniprot_id"] = ["QAAA", "QBBB", "P00000", "QAAA"]
        with pytest.raises(KeyError) as exc_info:
            preprocess_activities_for_eval(df, data_dir)
        msg = str(exc_info.value)
        assert "QAAA" in msg
        assert "QBBB" in msg

    def test_oov_mapping_resolves_unknown_target(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path, n_targets=5)
        oov = {"QUNKNWN": "P00000"}
        (data_dir / "oov_target_mapping.json").write_text(json.dumps(oov))
        df = _make_activities_df(n_rows=3)
        df = df.copy()
        df["uniprot_id"] = "QUNKNWN"
        # Should not raise
        result = preprocess_activities_for_eval(df, data_dir)
        assert len(result) == 6

    def test_missing_oov_file_is_ok(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import preprocess_activities_for_eval

        data_dir = _make_preproc_dir(tmp_path)
        assert not (data_dir / "oov_target_mapping.json").exists()
        result = preprocess_activities_for_eval(_make_activities_df(), data_dir)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# _print_metrics
# ---------------------------------------------------------------------------


class TestPrintMetrics:
    def _make_df(self, n_rows: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "assay_chembl_id": [f"CHEMBL{i % 3}" for i in range(n_rows)],
            "standard_type": ["IC50"] * n_rows,
            "split": ["test"] * n_rows,
            "pchembl_value_filled": rng.uniform(4, 9, n_rows).astype(np.float32),
            "pred_pchembl": rng.uniform(4, 9, n_rows).astype(np.float32),
        })

    def test_runs_without_error(self, capsys) -> None:
        from bind_pred_baseline.evaluate import _print_metrics

        _print_metrics(self._make_df())
        captured = capsys.readouterr()
        assert "Pearson" in captured.out

    def test_prints_each_split(self, capsys) -> None:
        from bind_pred_baseline.evaluate import _print_metrics

        rng = np.random.default_rng(1)
        n = 30
        df = pd.DataFrame({
            "assay_chembl_id": [f"CHEMBL{i % 3}" for i in range(n)],
            "standard_type": ["IC50"] * n,
            "split": ["test"] * 15 + ["2024_not_novel"] * 15,
            "pchembl_value_filled": rng.uniform(4, 9, n).astype(np.float32),
            "pred_pchembl": rng.uniform(4, 9, n).astype(np.float32),
        })
        _print_metrics(df)
        out = capsys.readouterr().out
        assert "test" in out
        assert "2024_not_novel" in out

    def test_prints_overall_when_multiple_splits(self, capsys) -> None:
        from bind_pred_baseline.evaluate import _print_metrics

        rng = np.random.default_rng(2)
        n = 30
        df = pd.DataFrame({
            "assay_chembl_id": [f"CHEMBL{i % 3}" for i in range(n)],
            "standard_type": ["IC50"] * n,
            "split": ["test"] * 15 + ["2024_not_novel"] * 15,
            "pchembl_value_filled": rng.uniform(4, 9, n).astype(np.float32),
            "pred_pchembl": rng.uniform(4, 9, n).astype(np.float32),
        })
        _print_metrics(df)
        out = capsys.readouterr().out
        assert "overall" in out

    def test_no_overall_line_for_single_split(self, capsys) -> None:
        from bind_pred_baseline.evaluate import _print_metrics

        _print_metrics(self._make_df())
        out = capsys.readouterr().out
        assert "overall" not in out


# ---------------------------------------------------------------------------
# evaluate_splits (integration)
# ---------------------------------------------------------------------------


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
        from bind_pred_baseline.evaluate import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        assert out_csv.exists()

    def test_output_csv_has_required_columns(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import evaluate_splits, OUTPUT_COLUMNS

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        for col in OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_output_rows_match_requested_split(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        activities_df = pd.read_parquet(acts)
        expected_n = len(activities_df[activities_df["split"] == "test"])

        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        result = pd.read_csv(out_csv)
        assert len(result) == expected_n

    def test_multiple_splits_in_output(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test", "2024_not_novel"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert set(df["split"].unique()) == {"test", "2024_not_novel"}

    def test_pred_pchembl_is_numeric(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert pd.api.types.is_float_dtype(df["pred_pchembl"])
        assert df["pred_pchembl"].notna().all()

    def test_raises_when_no_rows_for_any_split(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        with pytest.raises(ValueError, match="No rows found"):
            evaluate_splits(ckpt, data_dir, acts, tgts, splits=["nonexistent_split"], output_csv=out_csv)

    def test_split_column_preserved_in_output(self, tmp_path: Path) -> None:
        from bind_pred_baseline.evaluate import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        df = pd.read_csv(out_csv)
        assert (df["split"] == "test").all()

    def test_prints_pearson_r(self, tmp_path: Path, capsys) -> None:
        from bind_pred_baseline.evaluate import evaluate_splits

        data_dir, ckpt, acts, tgts = self._setup(tmp_path)
        out_csv = tmp_path / "predictions.csv"
        evaluate_splits(ckpt, data_dir, acts, tgts, splits=["test"], output_csv=out_csv)
        out = capsys.readouterr().out
        assert "Pearson" in out
