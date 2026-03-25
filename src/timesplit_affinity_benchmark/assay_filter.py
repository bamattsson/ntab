"""Assay-level filtering for val/test splits of the timesplit affinity benchmark.

Filtering is applied per (assay_chembl_id, standard_type) group within each
specified split independently. When this filter is active, duplicate rows with
the same (assay_chembl_id, standard_type, ligand_chembl_id) are also removed
from the output — see filter_assay_types for details.
"""

import pandas as pd


def _filter_single_split(
    df_split: pd.DataFrame,
    assay_docs_df: pd.DataFrame,
    only_equal_relation: bool,
    min_cpd_per_assay: int,
    min_std: float,
    one_assay_per_doi: bool,
) -> pd.DataFrame:
    """Apply all filtering steps to a single split's rows.

    Args:
        df_split: Rows belonging to one split value.
        assay_docs_df: DOI lookup table.
        only_equal_relation: See filter_assay_types.
        min_cpd_per_assay: See filter_assay_types.
        min_std: See filter_assay_types.
        one_assay_per_doi: See filter_assay_types.

    Returns:
        Filtered rows for this split.
    """
    # Step 1: drop non-"=" rows from the output
    if only_equal_relation:
        df_split = df_split[df_split["pchembl_relation"] == "="]

    # Step 2: deduplicate within each (assay, standard_type)
    df_split = df_split.drop_duplicates(
        subset=["assay_chembl_id", "standard_type", "ligand_chembl_id"]
    )

    if df_split.empty:
        return df_split

    # Steps 3 & 4: compute per-group stats and apply thresholds
    group_stats = (
        df_split
        .groupby(["assay_chembl_id", "standard_type"])
        .agg(
            n_cpd=("ligand_chembl_id", "nunique"),
            std=("pchembl_value_filled", "std"),
        )
        .reset_index()
    )
    passing = group_stats[
        (group_stats["n_cpd"] >= min_cpd_per_assay)
        & (group_stats["std"] >= min_std)
    ][["assay_chembl_id", "standard_type", "n_cpd"]].copy()

    if passing.empty:
        return df_split.iloc[:0]  # empty with same columns

    # Step 5: one assay per DOI
    if one_assay_per_doi:
        doi_map = (
            assay_docs_df[["assay_chembl_id", "doi"]]
            .drop_duplicates("assay_chembl_id")
        )
        passing = passing.merge(doi_map, on="assay_chembl_id", how="left")

        # Assays absent from assay_docs get NaN doi; treat each as its own group
        # by falling back to the assay ID so they are never collapsed together.
        passing["doi"] = passing["doi"].fillna(passing["assay_chembl_id"])

        passing["_assay_num"] = (
            passing["assay_chembl_id"]
            .str.replace("CHEMBL", "", regex=False)
            .astype(int)
        )

        # Sort so that the first row per DOI group is the winner
        passing = (
            passing
            .sort_values(["n_cpd", "_assay_num"], ascending=[False, True])
            .groupby("doi", dropna=False)
            .first()
            .reset_index()[["assay_chembl_id", "standard_type"]]
        )
    else:
        passing = passing[["assay_chembl_id", "standard_type"]]

    return df_split.merge(passing, on=["assay_chembl_id", "standard_type"], how="inner")


def filter_assay_types(
    activities_df: pd.DataFrame,
    assay_docs_df: pd.DataFrame,
    apply_to: list[str],
    only_equal_relation: bool,
    min_cpd_per_assay: int,
    min_std: float,
    one_assay_per_doi: bool,
) -> pd.DataFrame:
    """Filter activities to retain only well-characterised (assay, standard_type) groups.

    Each split listed in ``apply_to`` is filtered independently; thresholds are
    evaluated within each split separately so that a compound appearing in
    multiple splits does not inflate the count for any individual split.
    All other rows pass through unchanged.

    Within each affected split the following steps are applied in order:

    1. If ``only_equal_relation`` is True, rows where ``pchembl_relation != '='``
       are removed from the output.
    2. Duplicate rows sharing the same (assay_chembl_id, standard_type,
       ligand_chembl_id) are dropped.  This deduplication is applied to the
       output, not only used for threshold calculations.
    3. (assay, standard_type) groups with fewer than ``min_cpd_per_assay``
       unique compounds are removed.
    4. (assay, standard_type) groups whose ``pchembl_value_filled`` SD is below
       ``min_std`` are removed.
    5. If ``one_assay_per_doi`` is True, at most one (assay, standard_type) is
       kept per DOI within that split: the group with the most compounds, with
       ties broken by the lowest numeric part of the assay CHEMBL ID.  Assays
       absent from assay_docs_df (no DOI) are treated as their own DOI group and
       are never removed by this step.

    Args:
        activities_df: Full activities DataFrame containing a ``split`` column.
        assay_docs_df: DataFrame with columns ``assay_chembl_id`` and ``doi``,
            used for the one-assay-per-DOI filter.
        apply_to: Split names to filter (e.g. ``["test", "val_novel"]``).
        only_equal_relation: If True, drop rows where ``pchembl_relation != '='``
            from the affected splits before applying any other filter.
        min_cpd_per_assay: Minimum number of unique compounds required for an
            (assay, standard_type) group to be retained (inclusive).
        min_std: Minimum ``pchembl_value_filled`` standard deviation required
            for an (assay, standard_type) group to be retained (inclusive).
        one_assay_per_doi: If True, keep at most one (assay, standard_type) per
            DOI within each split independently.

    Returns:
        A new DataFrame with the filtered splits replacing the originals.
        Row order within unaffected splits is preserved; filtered split rows
        are appended at the end.
    """
    mask_affected = activities_df["split"].isin(apply_to)
    df_unaffected = activities_df[~mask_affected].copy()

    filtered_splits = []
    for split in apply_to:
        df_split = activities_df[activities_df["split"] == split].copy()
        if df_split.empty:
            continue
        filtered_splits.append(
            _filter_single_split(
                df_split=df_split,
                assay_docs_df=assay_docs_df,
                only_equal_relation=only_equal_relation,
                min_cpd_per_assay=min_cpd_per_assay,
                min_std=min_std,
                one_assay_per_doi=one_assay_per_doi,
            )
        )

    parts = [df_unaffected] + filtered_splits
    return pd.concat(parts, ignore_index=True)
