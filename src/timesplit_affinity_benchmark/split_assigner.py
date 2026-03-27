import pandas as pd


def assign_splits(
    activities_df: pd.DataFrame,
    compounds_df: pd.DataFrame,
    year_val_start: int = 2022,
    year_test_start: int = 2023,
) -> pd.DataFrame:
    """Assign a split label to each activity row based on doc_year and novelty.

    Split logic (with default year_val_start=2022, year_test_start=2023):
    - doc_year < year_val_start                                                    → "train"
    - year_val_start <= doc_year < year_test_start and is_novel == True            → "val_novel"
    - year_val_start <= doc_year < year_test_start and is_novel != True (or NaN)   → "val_not_novel"
    - doc_year >= year_test_start and is_novel == True                             → "test"
    - doc_year >= year_test_start and is_novel != True (or NaN)                   → "discard_not_novel"
    - doc_year is null                                                              → None

    Novelty is read from compounds_df columns named ``is_novel_{year_val_start}``
    and ``is_novel_{year_test_start}``, as produced by
    ``compute_novelty_for_cutoff``.  Reference-set compounds (cpd_earliest_year
    < cutoff, where cpd_earliest_year is MIN(year) across all ChEMBL activities
    via the activities → assays → docs path) have NaN for their novelty columns
    and are treated as not novel.

    Args:
        activities_df: Activity data with at least columns:
            - ligand_chembl_id
            - doc_year (numeric, nullable)
        compounds_df: DataFrame indexed by chembl_id with columns:
            - is_novel_{year_val_start} (bool or pd.NA)
            - is_novel_{year_test_start} (bool or pd.NA)
            Produced by merging two compute_novelty_for_cutoff outputs.
        year_val_start: First doc_year included in val; everything earlier is train.
        year_test_start: First doc_year included in test; must be > year_val_start.

    Returns:
        activities_df with a "split" column added (str, nullable).
    """
    result = activities_df.copy()
    year = result["doc_year"]
    split = pd.Series(index=result.index, dtype=object)

    split[year < year_val_start] = "train"

    mask_val = (year >= year_val_start) & (year < year_test_start)
    if mask_val.any():
        # .eq(True) returns a proper bool series: True → True, False/NA → False
        is_novel_val = result["ligand_chembl_id"].map(
            compounds_df[f"is_novel_{year_val_start}"]
        ).eq(True)
        split[mask_val & is_novel_val] = "val_novel"
        split[mask_val & ~is_novel_val] = "val_not_novel"

    mask_test = year >= year_test_start
    if mask_test.any():
        is_novel_test = result["ligand_chembl_id"].map(
            compounds_df[f"is_novel_{year_test_start}"]
        ).eq(True)
        split[mask_test & is_novel_test] = "test"
        split[mask_test & ~is_novel_test] = "discard_not_novel"

    split = split.where(split.notna(), other=None)
    result["split"] = split
    return result
