"""Tests for nfab.cluster_split.assign_cluster_splits."""

import numpy as np
import pandas as pd
import pytest

from nfab.cluster_split import assign_cluster_splits


def _make_activities(ligand_ids: list[str]) -> pd.DataFrame:
    """Minimal activities DataFrame with just ligand_chembl_id."""
    return pd.DataFrame({"ligand_chembl_id": ligand_ids})


def _make_clusters(chembl_ids: list[str], cluster_ids: list[int]) -> pd.DataFrame:
    """Minimal compound-cluster mapping DataFrame."""
    return pd.DataFrame({"chembl_id": chembl_ids, "cluster": cluster_ids})


def test_all_cluster_members_in_same_split() -> None:
    """All activities for compounds in the same cluster must land in the same split."""
    # 3 clusters of 5 compounds each
    activities = _make_activities(
        [f"C{cid}_M{m}" for cid in range(3) for m in range(5)]
    )
    clusters = _make_clusters(
        [f"C{cid}_M{m}" for cid in range(3) for m in range(5)],
        [cid for cid in range(3) for _ in range(5)],
    )
    result = assign_cluster_splits(activities, clusters, seed=42)
    for cid in range(3):
        mask = result["ligand_chembl_id"].str.startswith(f"C{cid}_")
        assert result.loc[mask, "split"].nunique() == 1, (
            f"Cluster {cid} has members in multiple splits"
        )


def test_datapoint_fractions_approximately_correct() -> None:
    """Test and val should contain approximately the requested fraction of datapoints."""
    rng = np.random.default_rng(0)
    n_clusters = 100
    # Assign 10 activities per cluster for uniform cluster sizes
    ligand_ids = [f"mol_{c}_{i}" for c in range(n_clusters) for i in range(10)]
    cluster_ids = [c for c in range(n_clusters) for _ in range(10)]
    activities = _make_activities(ligand_ids)
    clusters = _make_clusters(ligand_ids, cluster_ids)

    result = assign_cluster_splits(
        activities, clusters, val_frac=0.1, test_frac=0.1, seed=42
    )
    counts = result["split"].value_counts()
    total = len(result)
    assert 0.07 <= counts["test"] / total <= 0.15
    assert 0.07 <= counts["val"] / total <= 0.15
    assert counts["train"] / total >= 0.70


def test_output_split_labels_are_train_val_test() -> None:
    """Output split column must contain only 'train', 'val', 'test'."""
    activities = _make_activities([f"mol{i}" for i in range(30)])
    clusters = _make_clusters([f"mol{i}" for i in range(30)], list(range(30)))
    result = assign_cluster_splits(activities, clusters, seed=0)
    assert set(result["split"].unique()).issubset({"train", "val", "test"})
    assert "val" in result["split"].values
    assert "test" in result["split"].values
    assert "train" in result["split"].values


def test_deterministic_with_same_seed() -> None:
    """Same seed must produce identical split assignments."""
    activities = _make_activities([f"mol{i}" for i in range(50)])
    clusters = _make_clusters([f"mol{i}" for i in range(50)], list(range(50)))
    r1 = assign_cluster_splits(activities, clusters, seed=7)
    r2 = assign_cluster_splits(activities, clusters, seed=7)
    pd.testing.assert_frame_equal(r1, r2)


def test_raises_if_split_column_exists() -> None:
    """Should raise ValueError when activities_df already has a 'split' column."""
    activities = pd.DataFrame({"ligand_chembl_id": ["A"], "split": ["train"]})
    clusters = _make_clusters(["A"], [1])
    with pytest.raises(ValueError, match="split"):
        assign_cluster_splits(activities, clusters)


def test_raises_if_fracs_too_large() -> None:
    """Should raise ValueError when val_frac + test_frac >= 1.0."""
    activities = _make_activities(["A"])
    clusters = _make_clusters(["A"], [1])
    with pytest.raises(ValueError):
        assign_cluster_splits(activities, clusters, val_frac=0.5, test_frac=0.5)
    with pytest.raises(ValueError):
        assign_cluster_splits(activities, clusters, val_frac=0.6, test_frac=0.5)


def test_unmatched_compounds_raise_value_error() -> None:
    """Ligands absent from compounds_cluster_df must raise ValueError."""
    activities = _make_activities(["known", "unknown"])
    clusters = _make_clusters(["known"], [1])
    with pytest.raises(ValueError, match="no cluster assignment"):
        assign_cluster_splits(activities, clusters)
