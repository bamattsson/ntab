import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from bind_pred_baseline.preprocess_utils import (
    average_duplicates,
    build_target_index,
    find_closest_training_targets,
    load_activities,
    load_split_from_file,
    resolve_target_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_activities(**overrides) -> pd.DataFrame:
    """Minimal activities DataFrame with all required columns and sensible defaults."""
    base = {
        "target_chembl_id": ["T1", "T1", "T2", "T2"],
        "assay_chembl_id":  ["A1", "A1", "A2", "A2"],
        "ligand_chembl_id": ["L1", "L2", "L1", "L2"],
        "standard_type":    ["IC50", "IC50", "IC50", "IC50"],
        "pchembl_relation": ["=",    "=",    "=",    "="],
        "pchembl_value_filled": [7.0, 6.0, 5.0, 4.0],
        "split":            ["train", "train", "val_novel", "val_novel"],
        "canonical_smiles": ["C",    "CC",   "C",    "CC"],
        "mutation":         [None, None, None, None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    fpath = path / "activities.parquet"
    df.to_parquet(fpath, index=False)
    return fpath


# ---------------------------------------------------------------------------
# load_activities
# ---------------------------------------------------------------------------

class TestLoadActivities:
    def test_keeps_ic50_ki_kd_rows(self, tmp_path: Path) -> None:
        df = _make_activities(
            standard_type=["IC50", "Ki", "Kd", "EC50"],
            pchembl_relation=["=", "=", "=", "="],
        )
        path = _write_parquet(df, tmp_path)
        result = load_activities(path)
        assert set(result["standard_type"].unique()) == {"IC50", "Ki", "Kd"}

    def test_filters_to_equal_relation_only(self, tmp_path: Path) -> None:
        df = _make_activities(
            standard_type=["IC50", "IC50", "IC50", "IC50"],
            pchembl_relation=["=", "<", ">", "="],
        )
        path = _write_parquet(df, tmp_path)
        result = load_activities(path)
        assert set(result["pchembl_relation"].unique()) == {"="}
        assert len(result) == 2

    def test_keeps_all_columns(self, tmp_path: Path) -> None:
        df = _make_activities()
        path = _write_parquet(df, tmp_path)
        result = load_activities(path)
        for col in df.columns:
            assert col in result.columns

    def test_empty_result_when_no_allowed_type_equal_rows(self, tmp_path: Path) -> None:
        df = _make_activities(
            standard_type=["EC50", "EC50", "EC50", "IC50"],
            pchembl_relation=["=", "=", "=", "<"],
        )
        path = _write_parquet(df, tmp_path)
        result = load_activities(path)
        assert len(result) == 0

    def test_filters_out_mutation_rows(self, tmp_path: Path) -> None:
        df = _make_activities(mutation=[None, "V600E", None, "UNDEFINED MUTATION"])
        path = _write_parquet(df, tmp_path)
        result = load_activities(path)
        assert len(result) == 2
        assert result["mutation"].isna().all()

    def test_wildtype_rows_kept(self, tmp_path: Path) -> None:
        df = _make_activities(mutation=[None, None, None, None])
        path = _write_parquet(df, tmp_path)
        result = load_activities(path)
        assert len(result) == 4

    def test_filters_out_nan_pchembl(self, tmp_path: Path) -> None:
        import math
        df = _make_activities(pchembl_value_filled=[7.0, float("nan"), 5.0, 4.0])
        path = _write_parquet(df, tmp_path)
        result = load_activities(path)
        assert len(result) == 3
        assert not any(math.isnan(v) for v in result["pchembl_value_filled"])


# ---------------------------------------------------------------------------
# average_duplicates
# ---------------------------------------------------------------------------

class TestAverageDuplicates:
    def test_averages_pchembl_per_target_ligand(self) -> None:
        df = pd.DataFrame({
            "target_chembl_id": ["T1", "T1", "T1"],
            "ligand_chembl_id": ["L1", "L1", "L1"],
            "standard_type":    ["IC50", "IC50", "IC50"],
            "assay_chembl_id":  ["A1", "A2", "A3"],
            "pchembl_value_filled": [6.0, 8.0, 7.0],
            "split": ["train", "train", "train"],
        })
        result = average_duplicates(df)
        assert len(result) == 1
        assert pytest.approx(result.iloc[0]["pchembl_value_filled"]) == 7.0

    def test_single_measurement_unchanged(self) -> None:
        df = pd.DataFrame({
            "target_chembl_id": ["T1"],
            "ligand_chembl_id": ["L1"],
            "standard_type":    ["IC50"],
            "assay_chembl_id":  ["A1"],
            "pchembl_value_filled": [6.5],
            "split": ["train"],
        })
        result = average_duplicates(df)
        assert len(result) == 1
        assert pytest.approx(result.iloc[0]["pchembl_value_filled"]) == 6.5

    def test_returns_one_row_per_target_ligand_type_triple(self) -> None:
        df = pd.DataFrame({
            "target_chembl_id": ["T1", "T1", "T2", "T2"],
            "ligand_chembl_id": ["L1", "L1", "L1", "L1"],
            "standard_type":    ["IC50", "IC50", "IC50", "IC50"],
            "assay_chembl_id":  ["A1", "A2", "A1", "A3"],
            "pchembl_value_filled": [6.0, 8.0, 5.0, 7.0],
            "split": ["train"] * 4,
        })
        result = average_duplicates(df)
        assert len(result) == 2
        assert set(zip(result["target_chembl_id"], result["ligand_chembl_id"])) == {
            ("T1", "L1"), ("T2", "L1")
        }

    def test_different_pairs_not_mixed(self) -> None:
        df = pd.DataFrame({
            "target_chembl_id": ["T1", "T2"],
            "ligand_chembl_id": ["L1", "L1"],
            "standard_type":    ["IC50", "IC50"],
            "assay_chembl_id":  ["A1", "A2"],
            "pchembl_value_filled": [6.0, 9.0],
            "split": ["train", "train"],
        })
        result = average_duplicates(df).set_index(["target_chembl_id", "ligand_chembl_id"])
        assert pytest.approx(result.loc[("T1", "L1"), "pchembl_value_filled"]) == 6.0
        assert pytest.approx(result.loc[("T2", "L1"), "pchembl_value_filled"]) == 9.0

    def test_preserves_all_columns_after_averaging(self) -> None:
        df = pd.DataFrame({
            "target_chembl_id": ["T1", "T1"],
            "ligand_chembl_id": ["L1", "L1"],
            "standard_type":    ["IC50", "IC50"],
            "assay_chembl_id":  ["A1", "A2"],
            "pchembl_value_filled": [6.0, 8.0],
            "split": ["train", "train"],
            "uniprot_id": ["U1", "U1"],
        })
        result = average_duplicates(df)
        assert len(result) == 1
        assert "uniprot_id" in result.columns
        assert result.iloc[0]["uniprot_id"] == "U1"
        assert "assay_chembl_id" in result.columns

    def test_non_train_rows_are_not_averaged(self) -> None:
        df = pd.DataFrame({
            "target_chembl_id": ["T1", "T1"],
            "ligand_chembl_id": ["L1", "L1"],
            "standard_type":    ["IC50", "IC50"],
            "assay_chembl_id":  ["A1", "A2"],
            "pchembl_value_filled": [6.0, 8.0],
            "split": ["val_novel", "val_novel"],
        })
        result = average_duplicates(df)
        assert len(result) == 2
        assert set(result["pchembl_value_filled"].tolist()) == {6.0, 8.0}

    def test_ic50_and_ki_for_same_target_ligand_averaged_separately(self) -> None:
        # IC50 and Ki for the same (T1, L1) should produce two rows, not one
        df = pd.DataFrame({
            "target_chembl_id": ["T1", "T1", "T1", "T1"],
            "ligand_chembl_id": ["L1", "L1", "L1", "L1"],
            "standard_type":    ["IC50", "IC50", "Ki", "Ki"],
            "assay_chembl_id":  ["A1", "A2", "A3", "A4"],
            "pchembl_value_filled": [6.0, 8.0, 5.0, 9.0],
            "split": ["train"] * 4,
        })
        result = average_duplicates(df)
        assert len(result) == 2
        ic50_row = result[result["standard_type"] == "IC50"].iloc[0]
        ki_row = result[result["standard_type"] == "Ki"].iloc[0]
        assert pytest.approx(ic50_row["pchembl_value_filled"]) == 7.0
        assert pytest.approx(ki_row["pchembl_value_filled"]) == 7.0


# ---------------------------------------------------------------------------
# load_split_from_file
# ---------------------------------------------------------------------------

class TestLoadSplitFromFile:
    def _write_split_csv(self, tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
        fpath = tmp_path / "split.csv"
        pd.DataFrame(rows, columns=["uniprot_id", "data_split"]).to_csv(fpath, index=False)
        return fpath

    def test_returns_correct_mapping(self, tmp_path: Path) -> None:
        path = self._write_split_csv(tmp_path, [("U1", "train"), ("U2", "val"), ("U3", "test")])
        result = load_split_from_file(path)
        assert result == {"U1": "train", "U2": "val", "U3": "test"}

    def test_all_split_values_preserved(self, tmp_path: Path) -> None:
        path = self._write_split_csv(tmp_path, [("U1", "train"), ("U2", "val"), ("U3", "test")])
        result = load_split_from_file(path)
        assert set(result.values()) == {"train", "val", "test"}

    def test_raises_on_missing_uniprot_id_column(self, tmp_path: Path) -> None:
        fpath = tmp_path / "bad.csv"
        pd.DataFrame({"wrong_col": ["U1"], "data_split": ["train"]}).to_csv(fpath, index=False)
        with pytest.raises(ValueError, match="uniprot_id"):
            load_split_from_file(fpath)

    def test_raises_on_missing_data_split_column(self, tmp_path: Path) -> None:
        fpath = tmp_path / "bad.csv"
        pd.DataFrame({"uniprot_id": ["U1"], "wrong_col": ["train"]}).to_csv(fpath, index=False)
        with pytest.raises(ValueError, match="data_split"):
            load_split_from_file(fpath)

    def test_empty_csv_returns_empty_dict(self, tmp_path: Path) -> None:
        fpath = tmp_path / "empty.csv"
        pd.DataFrame(columns=["uniprot_id", "data_split"]).to_csv(fpath, index=False)
        result = load_split_from_file(fpath)
        assert result == {}


# ---------------------------------------------------------------------------
# build_target_index
# ---------------------------------------------------------------------------

class TestBuildTargetIndex:
    def test_assigns_unique_integer_to_each_target(self) -> None:
        index = build_target_index(["T1", "T2", "T3"])
        assert set(index.keys()) == {"T1", "T2", "T3"}
        assert len(set(index.values())) == 3

    def test_indices_are_contiguous_from_zero(self) -> None:
        index = build_target_index(["T1", "T2", "T3"])
        assert set(index.values()) == {0, 1, 2}

    def test_duplicates_in_input_collapsed(self) -> None:
        index = build_target_index(["T1", "T1", "T2", "T2"])
        assert len(index) == 2

    def test_deterministic_across_calls(self) -> None:
        index_a = build_target_index(["T3", "T1", "T2"])
        index_b = build_target_index(["T3", "T1", "T2"])
        assert index_a == index_b


# ---------------------------------------------------------------------------
# resolve_target_ids
# ---------------------------------------------------------------------------

class TestResolveTargetIds:
    def _index(self) -> dict[str, int]:
        return build_target_index(["T1", "T2", "T3"])

    def test_known_targets_resolve_correctly(self) -> None:
        index = self._index()
        result = resolve_target_ids(["T1", "T2"], index)
        assert result == [index["T1"], index["T2"]]

    def test_raises_on_oov_target(self) -> None:
        index = self._index()
        with pytest.raises(KeyError, match="T_UNKNOWN"):
            resolve_target_ids(["T_UNKNOWN"], index)

    def test_mapping_remaps_test_target_to_train_target(self) -> None:
        index = self._index()
        mapping = {"T_TEST": "T1"}
        result = resolve_target_ids(["T_TEST"], index, mapping=mapping)
        assert result == [index["T1"]]

    def test_mapping_oov_after_remap_raises(self) -> None:
        index = self._index()
        mapping = {"T_TEST": "T_ALSO_UNKNOWN"}
        with pytest.raises(KeyError):
            resolve_target_ids(["T_TEST"], index, mapping=mapping)

    def test_mapping_is_optional_and_none_by_default(self) -> None:
        index = self._index()
        result = resolve_target_ids(["T1"], index, mapping=None)
        assert result == [index["T1"]]


# ---------------------------------------------------------------------------
# find_closest_training_targets
# ---------------------------------------------------------------------------

class TestFindClosestTrainingTargets:
    def test_identical_sequence_maps_to_itself(self) -> None:
        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR"],
            sequences={"OOV": seq, "TR": seq},
        )
        assert result == {"OOV": "TR"}

    def test_returns_most_similar_not_least(self) -> None:
        # OOV is identical to TR_A and completely different from TR_B
        seq = "ACDEFGHIKLM"
        result = find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR_A", "TR_B"],
            sequences={
                "OOV":  seq,
                "TR_A": seq,                  # identical
                "TR_B": "XXXXXXXXXXXXXXXX",   # no overlap
            },
        )
        assert result["OOV"] == "TR_A"

    def test_multiple_oov_ids_all_mapped(self) -> None:
        result = find_closest_training_targets(
            oov_ids=["OOV_1", "OOV_2"],
            train_ids=["TR_A", "TR_B"],
            sequences={
                "OOV_1": "AAAAAAAAAA",
                "OOV_2": "BBBBBBBBBB",
                "TR_A":  "AAAAAAAAAA",
                "TR_B":  "BBBBBBBBBB",
            },
        )
        assert set(result.keys()) == {"OOV_1", "OOV_2"}
        assert result["OOV_1"] == "TR_A"
        assert result["OOV_2"] == "TR_B"

    def test_empty_oov_list_returns_empty_dict(self) -> None:
        result = find_closest_training_targets(
            oov_ids=[],
            train_ids=["TR"],
            sequences={"TR": "ACDEF"},
        )
        assert result == {}

    def test_oov_with_no_sequence_still_gets_a_mapping(self) -> None:
        # OOV has no sequence → can't compute similarity → still maps to some training target
        result = find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR"],
            sequences={"OOV": None, "TR": "ACDEF"},
        )
        assert "OOV" in result
        assert result["OOV"] == "TR"

    def test_training_target_with_no_sequence_is_skipped(self) -> None:
        # TR_A has no sequence, TR_B is identical to OOV → should pick TR_B
        seq = "ACDEFGHIKLM"
        result = find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR_A", "TR_B"],
            sequences={"OOV": seq, "TR_A": None, "TR_B": seq},
        )
        assert result["OOV"] == "TR_B"

    def test_nonstandard_chars_replaced_with_glycine(self) -> None:
        # "U" (selenocysteine) is not in BLOSUM62; seq has it but should still align
        seq = "ACDEFGHIKLM"
        result = find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR"],
            sequences={"OOV": "UCDEFGHIKLM", "TR": seq},  # U is non-standard
        )
        assert "OOV" in result

    def test_nonstandard_chars_print_warning(self, capsys) -> None:
        seq = "ACDEFGHIKLM"
        find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR"],
            sequences={"OOV": "UCDEFGHIKLM", "TR": seq},
        )
        captured = capsys.readouterr()
        assert "OOV" in captured.out
        assert "non-standard" in captured.out.lower() or "sanitiz" in captured.out.lower()

    def test_min_train_datapoints_filters_ineligible_targets(self) -> None:
        # TR_A has only 1 datapoint (below threshold), TR_B has 100 → must pick TR_B
        seq = "ACDEFGHIKLM"
        result = find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR_A", "TR_B"],
            sequences={"OOV": seq, "TR_A": seq, "TR_B": seq},
            train_counts={"TR_A": 1, "TR_B": 100},
            min_train_datapoints=50,
        )
        assert result["OOV"] == "TR_B"

    def test_min_train_datapoints_zero_disables_filtering(self) -> None:
        seq = "ACDEFGHIKLM"
        result = find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR_A", "TR_B"],
            sequences={"OOV": seq, "TR_A": seq, "TR_B": "XXXXXXXXXXX"},
            train_counts={"TR_A": 1, "TR_B": 100},
            min_train_datapoints=0,
        )
        assert result["OOV"] == "TR_A"

    def test_min_train_datapoints_fallback_when_none_qualify(self, capsys) -> None:
        # All train targets are below threshold → falls back to all targets, prints warning
        seq = "ACDEFGHIKLM"
        result = find_closest_training_targets(
            oov_ids=["OOV"],
            train_ids=["TR_A"],
            sequences={"OOV": seq, "TR_A": seq},
            train_counts={"TR_A": 1},
            min_train_datapoints=100,
        )
        assert "OOV" in result
        captured = capsys.readouterr()
        assert "falling back" in captured.out.lower()

    def test_parallel_matches_serial(self) -> None:
        result_serial = find_closest_training_targets(
            oov_ids=["OOV_1", "OOV_2"],
            train_ids=["TR_A", "TR_B"],
            sequences={
                "OOV_1": "AAAAAAAAAA",
                "OOV_2": "CCCCCCCCCC",
                "TR_A":  "AAAAAAAAAA",
                "TR_B":  "CCCCCCCCCC",
            },
            n_jobs=1,
        )
        result_parallel = find_closest_training_targets(
            oov_ids=["OOV_1", "OOV_2"],
            train_ids=["TR_A", "TR_B"],
            sequences={
                "OOV_1": "AAAAAAAAAA",
                "OOV_2": "CCCCCCCCCC",
                "TR_A":  "AAAAAAAAAA",
                "TR_B":  "CCCCCCCCCC",
            },
            n_jobs=2,
        )
        assert result_serial == result_parallel
