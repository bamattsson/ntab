import numpy as np
import pandas as pd


def assign_splits(
    activities_df: pd.DataFrame,
    is_novel_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign a split label to each activity row based on doc_year and novelty.

    Split logic:
    - doc_year < 2023                          → "train"
    - doc_year == 2023                         → "val"
    - doc_year >= 2024 and is_novel == True    → "test"
    - doc_year >= 2024 and is_novel == False   → "2024_not_novel"
    - doc_year is null                         → None

    Args:
        activities_df: Activity data with at least columns:
            - ligand_chembl_id
            - doc_year (numeric, nullable)
        is_novel_df: DataFrame indexed by chembl_id, covering all 2024+ compounds.
            Expected columns from filter_by_tanimoto: is_novel, max_similarity,
            most_similar_id. Built by the caller before calling this function.

    Returns:
        activities_df with a "split" column added (str, nullable).
    """
    result = activities_df.copy()

    year = result["doc_year"]
    split = pd.Series(index=result.index, dtype=object)

    split[year < 2023] = "train"
    split[year == 2023] = "val"

    mask_2024 = year >= 2024
    if mask_2024.any() and len(is_novel_df) > 0:
        is_novel = result["ligand_chembl_id"].map(is_novel_df["is_novel"])
        split[mask_2024 & (is_novel == True)] = "test"
        split[mask_2024 & (is_novel == False)] = "2024_not_novel"

    # null doc_year rows remain None (object NaN → replace with Python None)
    split = split.where(split.notna(), other=None)

    result["split"] = split
    return result
