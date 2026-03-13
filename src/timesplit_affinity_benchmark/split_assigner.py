import pandas as pd


def assign_splits(
    activities_df: pd.DataFrame,
    compounds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign a split label to each activity row based on doc_year and novelty.

    Split logic:
    - doc_year < 2023                                       → "train"
    - doc_year == 2023 and is_novel_2023 == True           → "val_novel"
    - doc_year == 2023 and is_novel_2023 != True (or NaN)  → "val_not_novel"
    - doc_year >= 2024 and is_novel_2024 == True           → "test"
    - doc_year >= 2024 and is_novel_2024 != True (or NaN)  → "2024_not_novel"
    - doc_year is null                                      → None

    Reference-set compounds (cpd_earliest_year < cutoff) have NaN for their novelty
    columns and are treated as not novel.

    Args:
        activities_df: Activity data with at least columns:
            - ligand_chembl_id
            - doc_year (numeric, nullable)
        compounds_df: DataFrame indexed by chembl_id with columns:
            - is_novel_2024 (bool or pd.NA)
            - is_novel_2023 (bool or pd.NA)
            Produced by merging two compute_novelty_for_cutoff outputs.

    Returns:
        activities_df with a "split" column added (str, nullable).
    """
    result = activities_df.copy()
    year = result["doc_year"]
    split = pd.Series(index=result.index, dtype=object)

    split[year < 2023] = "train"

    mask_2023 = year == 2023
    if mask_2023.any():
        # .eq(True) returns a proper bool series: True → True, False/NA → False
        is_novel_2023 = result["ligand_chembl_id"].map(compounds_df["is_novel_2023"]).eq(True)
        split[mask_2023 & is_novel_2023] = "val_novel"
        split[mask_2023 & ~is_novel_2023] = "val_not_novel"

    mask_2024 = year >= 2024
    if mask_2024.any():
        is_novel_2024 = result["ligand_chembl_id"].map(compounds_df["is_novel_2024"]).eq(True)
        split[mask_2024 & is_novel_2024] = "test"
        split[mask_2024 & ~is_novel_2024] = "2024_not_novel"

    split = split.where(split.notna(), other=None)
    result["split"] = split
    return result
