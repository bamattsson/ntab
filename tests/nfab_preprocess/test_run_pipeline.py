import pandas as pd

from nfab_preprocess.config import SimilarityBin
from nfab_preprocess.run_pipeline import assign_splits


YEAR_VAL = 2022
YEAR_TEST = 2023

TEST_BINS = [
    SimilarityBin(low=0.0, hi=0.35),
    SimilarityBin(low=0.35, hi=0.5),
    SimilarityBin(equal=1.0),
]


def _make_compounds_df(
    chembl_ids: list[str],
    sim_test: list[float],
    sim_val: list[float],
) -> pd.DataFrame:
    """Build a compounds_df as run_pipeline produces after compute_similarity_for_cutoff_year."""
    return pd.DataFrame(
        {
            f"max_sim_pre_{YEAR_TEST}": sim_test,
            f"most_sim_cpd_pre_{YEAR_TEST}": [None] * len(chembl_ids),
            f"max_sim_pre_{YEAR_VAL}": sim_val,
            f"most_sim_cpd_pre_{YEAR_VAL}": [None] * len(chembl_ids),
        },
        index=pd.Index(chembl_ids, name="chembl_id"),
    )


def test_train_rows() -> None:
    """doc_year < year_val_start → 'train' regardless of similarity."""
    activities = pd.DataFrame(
        {"ligand_chembl_id": ["A", "B"], "doc_year": [2019, 2021]}
    )
    compounds = _make_compounds_df(["A", "B"], [0.1, 0.9], [0.1, 0.9])
    result = assign_splits(activities, compounds, YEAR_VAL, YEAR_TEST, TEST_BINS)
    assert result["split"].tolist() == ["train", "train"]


def test_null_doc_year() -> None:
    """doc_year is null → None."""
    activities = pd.DataFrame({"ligand_chembl_id": ["A"], "doc_year": [None]})
    compounds = _make_compounds_df(["A"], [0.1], [0.1])
    result = assign_splits(activities, compounds, YEAR_VAL, YEAR_TEST, TEST_BINS)
    assert result["split"].iloc[0] is None


def test_test_range_bin() -> None:
    """Test-year rows land in the correct range bin."""
    activities = pd.DataFrame(
        {
            "ligand_chembl_id": ["A", "B"],
            "doc_year": [YEAR_TEST, YEAR_TEST],
        }
    )
    compounds = _make_compounds_df(["A", "B"], [0.2, 0.4], [0.0, 0.0])
    result = assign_splits(activities, compounds, YEAR_VAL, YEAR_TEST, TEST_BINS)
    splits = result.set_index("ligand_chembl_id")["split"].to_dict()
    assert splits["A"] == "test_sim_0.00_0.35"
    assert splits["B"] == "test_sim_0.35_0.50"


def test_test_exact_bin() -> None:
    """Test-year row with sim==1.0 lands in the equal bin."""
    activities = pd.DataFrame({"ligand_chembl_id": ["A"], "doc_year": [YEAR_TEST]})
    compounds = _make_compounds_df(["A"], [1.0], [0.0])
    result = assign_splits(activities, compounds, YEAR_VAL, YEAR_TEST, TEST_BINS)
    assert result["split"].iloc[0] == "test_sim_1.00"


def test_test_sim_not_in_any_bin_gets_none() -> None:
    """A test-year compound whose sim doesn't match any bin → None."""
    activities = pd.DataFrame({"ligand_chembl_id": ["A"], "doc_year": [YEAR_TEST]})
    # sim=0.8 is not in [0,0.35), [0.35,0.5), or =1.0
    compounds = _make_compounds_df(["A"], [0.8], [0.0])
    result = assign_splits(activities, compounds, YEAR_VAL, YEAR_TEST, TEST_BINS)
    assert result["split"].iloc[0] is None


def test_val_split_val_like_test_true() -> None:
    """Val-year rows get val_sim_* labels when split_val_like_test=True."""
    activities = pd.DataFrame(
        {"ligand_chembl_id": ["A", "B"], "doc_year": [YEAR_VAL, YEAR_VAL]}
    )
    compounds = _make_compounds_df(["A", "B"], [0.0, 0.0], [0.1, 1.0])
    result = assign_splits(
        activities,
        compounds,
        YEAR_VAL,
        YEAR_TEST,
        TEST_BINS,
        split_val_like_test=True,
    )
    splits = result.set_index("ligand_chembl_id")["split"].to_dict()
    assert splits["A"] == "val_sim_0.00_0.35"
    assert splits["B"] == "val_sim_1.00"


def test_val_split_val_like_test_false() -> None:
    """Val-year rows get 'val' when split_val_like_test=False."""
    activities = pd.DataFrame(
        {"ligand_chembl_id": ["A", "B"], "doc_year": [YEAR_VAL, YEAR_VAL]}
    )
    compounds = _make_compounds_df(["A", "B"], [0.0, 0.0], [0.1, 0.9])
    result = assign_splits(
        activities,
        compounds,
        YEAR_VAL,
        YEAR_TEST,
        TEST_BINS,
        split_val_like_test=False,
    )
    assert result["split"].tolist() == ["val", "val"]


def test_repeated_compound_gets_consistent_split() -> None:
    """A compound appearing in multiple rows always gets the same split label."""
    activities = pd.DataFrame(
        {
            "ligand_chembl_id": ["mol_A", "mol_A", "mol_A"],
            "doc_year": [YEAR_TEST, YEAR_TEST, YEAR_TEST],
        }
    )
    compounds = _make_compounds_df(["mol_A"], [0.2], [0.0])
    result = assign_splits(activities, compounds, YEAR_VAL, YEAR_TEST, TEST_BINS)
    assert result["split"].nunique() == 1
    assert result["split"].iloc[0] == "test_sim_0.00_0.35"


def test_range_bin_boundary_exclusive_hi() -> None:
    """The upper bound of a range bin is exclusive: sim==hi falls into the next bin."""
    activities = pd.DataFrame({"ligand_chembl_id": ["A"], "doc_year": [YEAR_TEST]})
    # sim=0.35 should land in [0.35, 0.5), not [0, 0.35)
    compounds = _make_compounds_df(["A"], [0.35], [0.0])
    result = assign_splits(activities, compounds, YEAR_VAL, YEAR_TEST, TEST_BINS)
    assert result["split"].iloc[0] == "test_sim_0.35_0.50"
