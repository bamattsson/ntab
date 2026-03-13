import numpy as np
import pandas as pd
import pytest

from timesplit_affinity_benchmark.split_assigner import assign_splits


def _make_compounds_df(
    chembl_ids: list[str],
    is_novel_2024: list,
    is_novel_2023: list,
) -> pd.DataFrame:
    """Build a compounds_df as compute_novelty_for_cutoff would produce it."""
    return pd.DataFrame(
        {
            "is_novel_2024": is_novel_2024,
            "max_sim_pre_2024": [0.0] * len(chembl_ids),
            "most_sim_cpd_pre_2024": [None] * len(chembl_ids),
            "is_novel_2023": is_novel_2023,
            "max_sim_pre_2023": [0.0] * len(chembl_ids),
            "most_sim_cpd_pre_2023": [None] * len(chembl_ids),
        },
        index=pd.Index(chembl_ids, name="chembl_id"),
    )


def test_train_and_null_splits() -> None:
    """train and None are assigned purely from doc_year; no 2023+ rows needed."""
    activities = pd.DataFrame({
        "ligand_chembl_id": ["A", "B", "C"],
        "doc_year": [2020, 2022, None],
    })
    compounds = _make_compounds_df(["A", "B", "C"], [pd.NA]*3, [pd.NA]*3)
    result = assign_splits(activities, compounds)
    assert result["split"].tolist() == ["train", "train", None]


def test_val_novel_and_val_not_novel() -> None:
    """2023 compounds are split into val_novel / val_not_novel based on is_novel_2023."""
    activities = pd.DataFrame({
        "ligand_chembl_id": ["train_mol", "novel_val", "similar_val"],
        "doc_year": [2020, 2023, 2023],
    })
    compounds = _make_compounds_df(
        chembl_ids=["train_mol", "novel_val", "similar_val"],
        is_novel_2024=[pd.NA, pd.NA, pd.NA],
        is_novel_2023=[pd.NA, True, False],
    )
    result = assign_splits(activities, compounds)
    splits = result.set_index("ligand_chembl_id")["split"].to_dict()
    assert splits["train_mol"] == "train"
    assert splits["novel_val"] == "val_novel"
    assert splits["similar_val"] == "val_not_novel"


def test_test_and_test_not_novel() -> None:
    """2024+ compounds are split into test / test_not_novel based on is_novel_2024."""
    activities = pd.DataFrame({
        "ligand_chembl_id": ["train_mol", "novel_test", "similar_test"],
        "doc_year": [2020, 2024, 2024],
    })
    compounds = _make_compounds_df(
        chembl_ids=["train_mol", "novel_test", "similar_test"],
        is_novel_2024=[pd.NA, True, False],
        is_novel_2023=[pd.NA, True, False],
    )
    result = assign_splits(activities, compounds)
    splits = result.set_index("ligand_chembl_id")["split"].to_dict()
    assert splits["train_mol"] == "train"
    assert splits["novel_test"] == "test"
    assert splits["similar_test"] == "2024_not_novel"


def test_nan_is_novel_treated_as_not_novel() -> None:
    """A 2024+ or 2023 compound with NaN is_novel (pre-filtered reference compound)
    is treated as not novel."""
    activities = pd.DataFrame({
        "ligand_chembl_id": ["old_in_test", "old_in_val"],
        "doc_year": [2024, 2023],
    })
    compounds = _make_compounds_df(
        chembl_ids=["old_in_test", "old_in_val"],
        is_novel_2024=[pd.NA, pd.NA],
        is_novel_2023=[pd.NA, pd.NA],
    )
    result = assign_splits(activities, compounds)
    splits = result.set_index("ligand_chembl_id")["split"].to_dict()
    assert splits["old_in_test"] == "2024_not_novel"
    assert splits["old_in_val"] == "val_not_novel"


def test_repeated_compound_gets_consistent_split() -> None:
    """A compound appearing in multiple activity rows always gets the same split label."""
    activities = pd.DataFrame({
        "ligand_chembl_id": ["mol_A", "mol_A", "mol_A"],
        "doc_year": [2024, 2024, 2024],
    })
    compounds = _make_compounds_df(
        chembl_ids=["mol_A"],
        is_novel_2024=[True],
        is_novel_2023=[True],
    )
    result = assign_splits(activities, compounds)
    assert result["split"].nunique() == 1
    assert result["split"].iloc[0] == "test"
