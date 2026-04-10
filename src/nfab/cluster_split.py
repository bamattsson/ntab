"""Cluster-based train/val/test split assignment for the affinity benchmark.

Compounds are split as whole clusters: every member of a cluster lands in the
same partition, ensuring no chemically similar compounds span train/val/test.

Splits are filled by datapoint count rather than cluster count: clusters are
shuffled randomly, then added one-by-one to test until the cumulative activity
count reaches ``test_frac * total``, then to val until ``val_frac * total`` is
reached. The remaining clusters are assigned to train.
"""

import numpy as np
import pandas as pd


def assign_cluster_splits(
    activities_df: pd.DataFrame,
    compounds_cluster_df: pd.DataFrame,
    cluster_col: str = "cluster",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits by filling clusters until datapoint fractions are met.

    Clusters are shuffled randomly (controlled by ``seed``), then added
    one-by-one to test until the cumulative activity count >=
    ``test_frac * total_activities``.  The same process then fills val.
    All remaining clusters go to train.

    Args:
        activities_df: Activity rows.  Must contain ``ligand_chembl_id``.
            Must NOT already have a ``split`` column.
        compounds_cluster_df: Compound-to-cluster mapping.  Must contain
            columns ``"chembl_id"`` and ``cluster_col``.
        cluster_col: Name of the cluster-ID column to use for splitting.
        val_frac: Target fraction of total activities to place in val
            (measured before assay-level quality filtering).
        test_frac: Target fraction of total activities to place in test
            (measured before assay-level quality filtering).
        seed: Random seed for the cluster shuffle.

    Returns:
        Copy of ``activities_df`` with a new ``"split"`` column containing
        ``"train"``, ``"val"``, or ``"test"``.  Row order is preserved.

    Raises:
        ValueError: If ``"split"`` already exists in ``activities_df``.
        ValueError: If ``val_frac + test_frac >= 1.0``.
        ValueError: If any ``ligand_chembl_id`` in ``activities_df`` is absent
            from ``compounds_cluster_df``.
            # If the clustered file ever changes, drop unmatched rows instead — we must be strict.
    """
    if "split" in activities_df.columns:
        raise ValueError(
            "'split' column already present in activities_df. "
            "Drop it before calling assign_cluster_splits."
        )
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac ({val_frac}) + test_frac ({test_frac}) must be < 1.0."
        )
    if cluster_col not in compounds_cluster_df.columns:
        raise ValueError(
            f"cluster_col '{cluster_col}' not found in compounds_cluster_df. "
            f"Available columns: {list(compounds_cluster_df.columns)}"
        )

    # Build ligand → cluster_id mapping
    cpd_map: pd.Series = (
        compounds_cluster_df[["chembl_id", cluster_col]]
        .drop_duplicates("chembl_id")
        .set_index("chembl_id")[cluster_col]
    )

    # Check for unmatched ligands
    all_ligands = activities_df["ligand_chembl_id"].unique()
    unmatched = [lid for lid in all_ligands if lid not in cpd_map.index]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} ligand(s) have no cluster assignment in "
            f"compounds_cluster_df. First few: {unmatched[:5]}"
        )

    # Map each activity row to its cluster_id
    ligand_cluster = activities_df["ligand_chembl_id"].map(cpd_map)

    # Count activities per cluster (for datapoint-fraction filling)
    total = len(activities_df)
    cluster_activity_counts: pd.Series = ligand_cluster.value_counts()

    # Shuffle clusters deterministically
    rng = np.random.default_rng(seed)
    shuffled_clusters = rng.permutation(
        sorted(cluster_activity_counts.index.tolist())
    ).tolist()

    # Fill test, then val, then train
    split_map: dict = {}
    cumulative = 0
    test_target = test_frac * total
    val_target = val_frac * total
    phase = "test"

    for cluster_id in shuffled_clusters:
        count = cluster_activity_counts[cluster_id]
        if phase == "test":
            split_map[cluster_id] = "test"
            cumulative += count
            if cumulative >= test_target:
                phase = "val"
                cumulative = 0
        elif phase == "val":
            split_map[cluster_id] = "val"
            cumulative += count
            if cumulative >= val_target:
                phase = "train"
        else:
            split_map[cluster_id] = "train"

    result = activities_df.copy()
    result["split"] = ligand_cluster.map(split_map)
    return result
