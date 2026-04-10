import pandas as pd

from nfab.run_pipeline import assign_splits


YEAR_VAL = 2022
YEAR_TEST = 2023


def _make_compounds_df(
    chembl_ids: list[str],
    is_novel_test: list,
    is_novel_val: list,
) -> pd.DataFrame:
    """Build a compounds_df as compute_novelty_for_cutoff would produce it."""
    return pd.DataFrame(
        {
            f"is_novel_{YEAR_TEST}": is_novel_test,
            f"max_sim_pre_{YEAR_TEST}": [0.0] * len(chembl_ids),
            f"most_sim_cpd_pre_{YEAR_TEST}": [None] * len(chembl_ids),
            f"is_novel_{YEAR_VAL}": is_novel_val,
            f"max_sim_pre_{YEAR_VAL}": [0.0] * len(chembl_ids),
            f"most_sim_cpd_pre_{YEAR_VAL}": [None] * len(chembl_ids),
        },
        index=pd.Index(chembl_ids, name="chembl_id"),
    )


def test_train_and_null_splits() -> None:
    """train and None are assigned purely from doc_year; rows before year_val_start are train."""
    activities = pd.DataFrame(
        {
            "ligand_chembl_id": ["A", "B", "C"],
            "doc_year": [2019, 2021, None],
        }
    )
    compounds = _make_compounds_df(["A", "B", "C"], [pd.NA] * 3, [pd.NA] * 3)
    result = assign_splits(
        activities, compounds, year_val_start=YEAR_VAL, year_test_start=YEAR_TEST
    )
    assert result["split"].tolist() == ["train", "train", None]


def test_val_novel_and_val_not_novel() -> None:
    """Val-year compounds are split into val_novel / val_not_novel based on compound novelty."""
    activities = pd.DataFrame(
        {
            "ligand_chembl_id": ["train_mol", "novel_val", "similar_val"],
            "doc_year": [2020, YEAR_VAL, YEAR_VAL],
        }
    )
    compounds = _make_compounds_df(
        chembl_ids=["train_mol", "novel_val", "similar_val"],
        is_novel_test=[pd.NA, pd.NA, pd.NA],
        is_novel_val=[pd.NA, True, False],
    )
    result = assign_splits(
        activities, compounds, year_val_start=YEAR_VAL, year_test_start=YEAR_TEST
    )
    splits = result.set_index("ligand_chembl_id")["split"].to_dict()
    assert splits["train_mol"] == "train"
    assert splits["novel_val"] == "val_novel"
    assert splits["similar_val"] == "val_not_novel"


def test_test_and_discard_not_novel() -> None:
    """Test-year compounds are split into test / discard_not_novel based on compound novelty."""
    activities = pd.DataFrame(
        {
            "ligand_chembl_id": ["train_mol", "novel_test", "similar_test"],
            "doc_year": [2020, YEAR_TEST, YEAR_TEST],
        }
    )
    compounds = _make_compounds_df(
        chembl_ids=["train_mol", "novel_test", "similar_test"],
        is_novel_test=[pd.NA, True, False],
        is_novel_val=[pd.NA, True, False],
    )
    result = assign_splits(
        activities, compounds, year_val_start=YEAR_VAL, year_test_start=YEAR_TEST
    )
    splits = result.set_index("ligand_chembl_id")["split"].to_dict()
    assert splits["train_mol"] == "train"
    assert splits["novel_test"] == "test"
    assert splits["similar_test"] == "discard_not_novel"


def test_nan_is_novel_treated_as_not_novel() -> None:
    """A test-year or val-year compound with NaN is_novel (pre-filtered reference compound)
    is treated as not novel."""
    activities = pd.DataFrame(
        {
            "ligand_chembl_id": ["old_in_test", "old_in_val"],
            "doc_year": [YEAR_TEST, YEAR_VAL],
        }
    )
    compounds = _make_compounds_df(
        chembl_ids=["old_in_test", "old_in_val"],
        is_novel_test=[pd.NA, pd.NA],
        is_novel_val=[pd.NA, pd.NA],
    )
    result = assign_splits(
        activities, compounds, year_val_start=YEAR_VAL, year_test_start=YEAR_TEST
    )
    splits = result.set_index("ligand_chembl_id")["split"].to_dict()
    assert splits["old_in_test"] == "discard_not_novel"
    assert splits["old_in_val"] == "val_not_novel"


def test_repeated_compound_gets_consistent_split() -> None:
    """A compound appearing in multiple activity rows always gets the same split label."""
    activities = pd.DataFrame(
        {
            "ligand_chembl_id": ["mol_A", "mol_A", "mol_A"],
            "doc_year": [YEAR_TEST, YEAR_TEST, YEAR_TEST],
        }
    )
    compounds = _make_compounds_df(
        chembl_ids=["mol_A"],
        is_novel_test=[True],
        is_novel_val=[True],
    )
    result = assign_splits(
        activities, compounds, year_val_start=YEAR_VAL, year_test_start=YEAR_TEST
    )
    assert result["split"].nunique() == 1
    assert result["split"].iloc[0] == "test"
