import numpy as np
import pandas as pd

from ntab_preprocess.assay_filter import filter_assay_types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_activities(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal activities DataFrame for testing."""
    defaults = {"pchembl_relation": "="}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_assay_docs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["assay_chembl_id", "doc_chembl_id"])


def _passing_values(n: int = 10) -> list[float]:
    """Return n pchembl values with SD well above 0.5."""
    return list(np.linspace(4.0, 8.0, n))


# ---------------------------------------------------------------------------
# Basic threshold filtering
# ---------------------------------------------------------------------------


def test_assay_below_min_cpd_is_removed() -> None:
    """An assay-type with fewer compounds than min_cpd_per_assay is dropped."""
    activities = _make_activities(
        [
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"CPD{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
            for i, v in enumerate(_passing_values(n=9))  # 9 < 10
        ]
    )
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert result.empty


def test_assay_meeting_min_cpd_exactly_is_kept() -> None:
    """An assay-type with exactly min_cpd_per_assay compounds is kept (>= is inclusive)."""
    activities = _make_activities(
        [
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"CPD{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
            for i, v in enumerate(_passing_values(n=10))
        ]
    )
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert len(result) == 10


def test_assay_below_min_std_is_removed() -> None:
    """An assay-type with SD below min_std is dropped."""
    activities = _make_activities(
        [
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"CPD{i}",
                "pchembl_value_filled": 5.0,  # zero variance
                "split": "test",
            }
            for i in range(10)
        ]
    )
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert result.empty


def test_assay_meeting_min_std_exactly_is_kept() -> None:
    """An assay-type with SD >= min_std is kept; one with SD just below is removed (>= is inclusive)."""
    values = _passing_values(n=10)
    actual_std = float(pd.Series(values).std())

    activities = _make_activities(
        [
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"CPD{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
            for i, v in enumerate(values)
        ]
    )
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    # Threshold just below actual SD: assay should pass
    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=actual_std - 0.01,
        one_assay_per_doc=False,
    )
    assert len(result) == 10

    # Threshold just above actual SD: assay should be removed
    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=actual_std + 0.01,
        one_assay_per_doc=False,
    )
    assert result.empty


def test_assay_passing_both_thresholds_is_kept() -> None:
    """An assay-type meeting both N and SD thresholds is fully retained."""
    activities = _make_activities(
        [
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"CPD{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
            for i, v in enumerate(_passing_values(n=10))
        ]
    )
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert len(result) == 10
    assert set(result["assay_chembl_id"]) == {"CHEMBL1"}


# ---------------------------------------------------------------------------
# Train rows are never touched
# ---------------------------------------------------------------------------


def test_train_rows_are_never_filtered() -> None:
    """Rows in splits not listed in apply_to pass through unchanged."""
    train_rows = [
        {
            "assay_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "ligand_chembl_id": f"CPD{i}",
            "pchembl_value_filled": 5.0,
            "split": "train",
        }
        for i in range(3)  # would fail N threshold if filtered
    ]
    activities = _make_activities(train_rows)
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert len(result) == 3
    assert (result["split"] == "train").all()


def test_apply_to_limits_which_splits_are_filtered() -> None:
    """Only splits listed in apply_to are filtered; others pass through unchanged."""
    rows = []
    # val_not_novel: failing assay (3 cpds) — should survive because not in apply_to
    for i in range(3):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"V{i}",
                "pchembl_value_filled": 5.0,
                "split": "val_not_novel",
            }
        )
    # test: failing assay — should be removed
    for i in range(3):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL2",
                "standard_type": "IC50",
                "ligand_chembl_id": f"T{i}",
                "pchembl_value_filled": 5.0,
                "split": "test",
            }
        )

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [
            {"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"},
            {"assay_chembl_id": "CHEMBL2", "doc_chembl_id": "CHEMBL1000002"},
        ]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )

    assert (result["split"] == "val_not_novel").all()
    assert len(result) == 3


# ---------------------------------------------------------------------------
# only_equal_relation
# ---------------------------------------------------------------------------


def test_only_equal_relation_removes_non_equal_rows_from_output() -> None:
    """When only_equal_relation=True, non-'=' rows are dropped from filtered splits."""
    rows = [
        {
            "assay_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "ligand_chembl_id": f"CPD{i}",
            "pchembl_value_filled": v,
            "pchembl_relation": "=",
            "split": "test",
        }
        for i, v in enumerate(_passing_values(n=10))
    ]
    # Add a non-"=" row that would otherwise count toward the compound total
    rows.append(
        {
            "assay_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "ligand_chembl_id": "CPD_CENSORED",
            "pchembl_value_filled": 7.0,
            "pchembl_relation": "<",
            "split": "test",
        }
    )

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=True,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )

    assert (result["pchembl_relation"] == "=").all()
    assert "CPD_CENSORED" not in result["ligand_chembl_id"].values


def test_only_equal_relation_false_keeps_non_equal_rows() -> None:
    """When only_equal_relation=False, non-'=' rows are kept in the output."""
    rows = [
        {
            "assay_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "ligand_chembl_id": f"CPD{i}",
            "pchembl_value_filled": v,
            "pchembl_relation": "=" if i < 9 else "<",
            "split": "test",
        }
        for i, v in enumerate(_passing_values(n=10))
    ]
    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )

    assert "<" in result["pchembl_relation"].values
    assert len(result) == 10


def test_only_equal_relation_non_equal_rows_count_toward_n_threshold() -> None:
    """Non-'=' rows are excluded BEFORE counting: assay with only 9 '=' rows fails min_cpd=10."""
    rows = [
        {
            "assay_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "ligand_chembl_id": f"CPD{i}",
            "pchembl_value_filled": v,
            "pchembl_relation": "=",
            "split": "test",
        }
        for i, v in enumerate(_passing_values(n=9))
    ]
    # This non-"=" row would push total to 10, but should be excluded before threshold check
    rows.append(
        {
            "assay_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "ligand_chembl_id": "CPD_EXTRA",
            "pchembl_value_filled": 7.0,
            "pchembl_relation": "<",
            "split": "test",
        }
    )

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=True,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert result.empty


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_duplicate_compound_rows_are_removed_from_output() -> None:
    """Duplicate (assay, standard_type, ligand) rows are dropped from the output."""
    rows = [
        {
            "assay_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "ligand_chembl_id": f"CPD{i}",
            "pchembl_value_filled": v,
            "split": "test",
        }
        for i, v in enumerate(_passing_values(n=10))
    ]
    # Duplicate the first compound
    rows.append({**rows[0]})

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert len(result) == 10


def test_deduplication_counts_unique_compounds_for_threshold() -> None:
    """Duplicates are removed before checking N threshold: 10 rows but only 9 unique cpds fails."""
    unique_rows = [
        {
            "assay_chembl_id": "CHEMBL1",
            "standard_type": "IC50",
            "ligand_chembl_id": f"CPD{i}",
            "pchembl_value_filled": v,
            "split": "test",
        }
        for i, v in enumerate(_passing_values(n=9))
    ]
    # Add a duplicate of CPD0 to bring total rows to 10, but unique cpds stays at 9
    unique_rows.append({**unique_rows[0]})

    activities = _make_activities(unique_rows)
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert result.empty


# ---------------------------------------------------------------------------
# one_assay_per_doc
# ---------------------------------------------------------------------------


def test_one_assay_per_doc_keeps_assay_with_most_compounds() -> None:
    """When two assay-types from the same DOI pass filters, the larger one is kept."""
    rows = []
    # CHEMBL1: 15 compounds
    for i, v in enumerate(_passing_values(n=15)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"A{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
        )
    # CHEMBL2: 10 compounds — same DOI, fewer compounds
    for i, v in enumerate(_passing_values(n=10)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL2",
                "standard_type": "IC50",
                "ligand_chembl_id": f"B{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
        )

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [
            {"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"},
            {"assay_chembl_id": "CHEMBL2", "doc_chembl_id": "CHEMBL1000001"},
        ]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=True,
    )

    assert set(result["assay_chembl_id"]) == {"CHEMBL1"}
    assert len(result) == 15


def test_one_assay_per_doc_tiebreaks_by_lowest_assay_number() -> None:
    """When two assay-types from the same DOI have equal compound counts, the lower CHEMBL ID wins."""
    rows = []
    for i, v in enumerate(_passing_values(n=10)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL100",
                "standard_type": "IC50",
                "ligand_chembl_id": f"A{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
        )
    for i, v in enumerate(_passing_values(n=10)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL200",
                "standard_type": "IC50",
                "ligand_chembl_id": f"B{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
        )

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [
            {"assay_chembl_id": "CHEMBL100", "doc_chembl_id": "CHEMBL1000001"},
            {"assay_chembl_id": "CHEMBL200", "doc_chembl_id": "CHEMBL1000001"},
        ]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=True,
    )

    assert set(result["assay_chembl_id"]) == {"CHEMBL100"}


def test_one_assay_per_doc_different_standard_types_same_doi_keeps_only_one() -> None:
    """Two assay-types with the same DOI but different standard_types: only one survives."""
    rows = []
    for i, v in enumerate(_passing_values(n=10)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"A{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
        )
    for i, v in enumerate(_passing_values(n=15)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL2",
                "standard_type": "Kd",
                "ligand_chembl_id": f"B{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
        )

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [
            {"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"},
            {"assay_chembl_id": "CHEMBL2", "doc_chembl_id": "CHEMBL1000001"},
        ]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=True,
    )

    # CHEMBL2/Kd has more compounds, should win
    assert set(result["assay_chembl_id"]) == {"CHEMBL2"}
    assert set(result["standard_type"]) == {"Kd"}


def test_one_assay_per_doc_assay_without_doc_is_kept() -> None:
    """Assays with no matching doc_chembl_id in assay_docs are treated as their own group and kept."""
    rows = [
        {
            "assay_chembl_id": "CHEMBL999",
            "standard_type": "IC50",
            "ligand_chembl_id": f"CPD{i}",
            "pchembl_value_filled": v,
            "split": "test",
        }
        for i, v in enumerate(_passing_values(n=10))
    ]
    docs = _make_assay_docs([])  # no doc mapping at all

    result = filter_assay_types(
        rows if False else _make_activities(rows),
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=True,
    )

    assert len(result) == 10
    assert set(result["assay_chembl_id"]) == {"CHEMBL999"}


def test_one_assay_per_doc_different_docs_both_kept() -> None:
    """Two assay-types from different documents both survive one_assay_per_doc."""
    rows = []
    for i, v in enumerate(_passing_values(n=10)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"A{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
        )
    for i, v in enumerate(_passing_values(n=10)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL2",
                "standard_type": "IC50",
                "ligand_chembl_id": f"B{i}",
                "pchembl_value_filled": v,
                "split": "test",
            }
        )

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [
            {"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"},
            {"assay_chembl_id": "CHEMBL2", "doc_chembl_id": "CHEMBL1000002"},
        ]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["test"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=True,
    )

    assert set(result["assay_chembl_id"]) == {"CHEMBL1", "CHEMBL2"}
    assert len(result) == 20


# ---------------------------------------------------------------------------
# Per-split independence
# ---------------------------------------------------------------------------


def test_thresholds_are_evaluated_per_split_not_across_splits() -> None:
    """An assay with 6 compounds in val_not_novel and 6 in val_novel (12 total)
    must NOT pass min_cpd=10, because each split is evaluated independently."""
    rows = []
    for i, v in enumerate(_passing_values(n=6)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"A{i}",
                "pchembl_value_filled": v,
                "split": "val_not_novel",
            }
        )
    for i, v in enumerate(_passing_values(n=6)):
        rows.append(
            {
                "assay_chembl_id": "CHEMBL1",
                "standard_type": "IC50",
                "ligand_chembl_id": f"B{i}",
                "pchembl_value_filled": v,
                "split": "val_novel",
            }
        )

    activities = _make_activities(rows)
    docs = _make_assay_docs(
        [{"assay_chembl_id": "CHEMBL1", "doc_chembl_id": "CHEMBL1000001"}]
    )

    result = filter_assay_types(
        activities,
        docs,
        apply_to=["val_not_novel", "val_novel"],
        only_equal_relation=False,
        min_cpd_per_assay=10,
        min_std=0.5,
        one_assay_per_doc=False,
    )
    assert result.empty
