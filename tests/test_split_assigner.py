import numpy as np
import pandas as pd
import pytest

from timesplit_affinity_benchmark.split_assigner import assign_splits


def _make_is_novel_df(chembl_ids: list[str], is_novel: list[bool]) -> pd.DataFrame:
    """Build is_novel_df as the caller would construct it from filter_by_tanimoto output."""
    return pd.DataFrame(
        {
            "is_novel": is_novel,
            "max_similarity": [0.0] * len(chembl_ids),
            "most_similar_id": [None] * len(chembl_ids),
        },
        index=pd.Index(chembl_ids, name="chembl_id"),
    )


def test_year_based_splits() -> None:
    """train/val/None are assigned purely from doc_year with no 2024+ rows."""
    activities = pd.DataFrame({
        "ligand_chembl_id": ["A", "B", "C", "D"],
        "doc_year": [2020, 2022, 2023, None],
    })
    result = assign_splits(activities, is_novel_df=pd.DataFrame())
    assert result["split"].tolist() == ["train", "train", "val", None]


def test_tanimoto_splits_2024() -> None:
    """2024+ compounds are split into test vs 2024_not_novel based on is_novel."""
    activities = pd.DataFrame({
        "ligand_chembl_id": ["train_mol", "novel_2024", "similar_2024"],
        "doc_year": [2020, 2024, 2024],
    })
    is_novel_df = _make_is_novel_df(
        chembl_ids=["novel_2024", "similar_2024"],
        is_novel=[True, False],
    )
    result = assign_splits(activities, is_novel_df)
    assert result.set_index("ligand_chembl_id")["split"].to_dict() == {
        "train_mol": "train",
        "novel_2024": "test",
        "similar_2024": "2024_not_novel",
    }


def test_repeated_compound_gets_consistent_split() -> None:
    """A compound appearing in multiple activity rows always gets the same split label."""
    activities = pd.DataFrame({
        "ligand_chembl_id": ["train_mol", "mol_A", "mol_A", "mol_A"],
        "doc_year": [2020, 2024, 2024, 2024],
    })
    is_novel_df = _make_is_novel_df(chembl_ids=["mol_A"], is_novel=[True])
    result = assign_splits(activities, is_novel_df)
    mol_a_splits = result[result["ligand_chembl_id"] == "mol_A"]["split"]
    assert mol_a_splits.nunique() == 1
    assert mol_a_splits.iloc[0] == "test"
