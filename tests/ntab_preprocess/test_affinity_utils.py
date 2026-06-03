import numpy as np
import pandas as pd
import pytest

from ntab_preprocess.affinity_utils import (
    add_pchembl_columns,
    compute_pchembl_filled,
    invert_relation,
)


# ---------------------------------------------------------------------------
# invert_relation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("=", "="),
        ("~", "~"),
        (">", "<"),
        (">=", "<="),
        (">>", "<<"),
        ("<", ">"),
        ("<=", ">="),
        ("<<", ">>"),
        (None, None),
    ],
)
def test_invert_relation_known_values(inp, expected):
    assert invert_relation(inp) == expected


def test_invert_relation_nan_returns_none():
    assert invert_relation(float("nan")) is None


def test_invert_relation_unknown_raises():
    with pytest.raises(ValueError):
        invert_relation("??")


# ---------------------------------------------------------------------------
# compute_pchembl_filled
# ---------------------------------------------------------------------------


def test_compute_pchembl_filled_1000nM():
    row = pd.Series({"standard_value": 1000.0, "standard_units": "nM"})
    result = compute_pchembl_filled(row)
    assert result == pytest.approx(6.0, abs=0.01)


def test_compute_pchembl_filled_non_nM_returns_nan():
    row = pd.Series({"standard_value": 1000.0, "standard_units": "uM"})
    assert np.isnan(compute_pchembl_filled(row))


def test_compute_pchembl_filled_none_value_returns_nan():
    row = pd.Series({"standard_value": None, "standard_units": "nM"})
    assert np.isnan(compute_pchembl_filled(row))


def test_compute_pchembl_filled_decimal_value():
    """Decimal type from psycopg2/PostgreSQL NUMERIC columns must be handled."""
    from decimal import Decimal

    row = pd.Series(
        {"standard_value": Decimal("20000.0000000"), "standard_units": "nM"}
    )
    result = compute_pchembl_filled(row)
    assert result == pytest.approx(4.70, abs=0.01)


# ---------------------------------------------------------------------------
# add_pchembl_columns
# ---------------------------------------------------------------------------


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_pchembl_value_filled_uses_existing_pchembl_value():
    """When pchembl_value is already present, use it directly."""
    df = _make_df(
        [
            {
                "pchembl_value": 7.5,
                "standard_value": 100.0,
                "standard_units": "nM",
                "standard_relation": "=",
            },
        ]
    )
    out = add_pchembl_columns(df)
    assert out["pchembl_value_filled"].iloc[0] == pytest.approx(7.5)


def test_pchembl_value_filled_computes_when_missing():
    """When pchembl_value is NaN (censored measurement), compute from standard_value."""
    df = _make_df(
        [
            {
                "pchembl_value": np.nan,
                "standard_value": 1000.0,
                "standard_units": "nM",
                "standard_relation": ">",
            },
        ]
    )
    out = add_pchembl_columns(df)
    assert out["pchembl_value_filled"].iloc[0] == pytest.approx(6.0, abs=0.01)


def test_pchembl_value_filled_stays_nan_for_non_nM():
    df = _make_df(
        [
            {
                "pchembl_value": np.nan,
                "standard_value": 1.0,
                "standard_units": "uM",
                "standard_relation": ">",
            },
        ]
    )
    out = add_pchembl_columns(df)
    assert np.isnan(out["pchembl_value_filled"].iloc[0])


def test_pchembl_relation_inverted():
    """standard_relation '>' becomes pchembl_relation '<'."""
    df = _make_df(
        [
            {
                "pchembl_value": np.nan,
                "standard_value": 1000.0,
                "standard_units": "nM",
                "standard_relation": ">",
            },
            {
                "pchembl_value": 7.0,
                "standard_value": None,
                "standard_units": "nM",
                "standard_relation": "=",
            },
        ]
    )
    out = add_pchembl_columns(df)
    assert out["pchembl_relation"].iloc[0] == "<"
    assert out["pchembl_relation"].iloc[1] == "="


def test_add_pchembl_columns_does_not_modify_original():
    df = _make_df(
        [
            {
                "pchembl_value": np.nan,
                "standard_value": 100.0,
                "standard_units": "nM",
                "standard_relation": ">",
            },
        ]
    )
    add_pchembl_columns(df)
    assert "pchembl_value_filled" not in df.columns
    assert "pchembl_value" in df.columns
    assert "standard_relation" in df.columns


def test_add_pchembl_columns_replaces_originals():
    """pchembl_value and standard_relation are dropped; new columns take their place."""
    df = _make_df(
        [
            {
                "pchembl_value": np.nan,
                "standard_value": 100.0,
                "standard_units": "nM",
                "standard_relation": ">",
            },
        ]
    )
    out = add_pchembl_columns(df)
    assert "pchembl_value" not in out.columns
    assert "standard_relation" not in out.columns
    assert "pchembl_value_filled" in out.columns
    assert "pchembl_relation" in out.columns


def test_add_pchembl_columns_preserves_column_order():
    """New columns appear at the same positions as the originals they replace."""
    df = _make_df(
        [
            {
                "a": 1,
                "pchembl_value": 7.0,
                "b": 2,
                "standard_relation": "=",
                "c": 3,
                "standard_value": 100.0,
                "standard_units": "nM",
            },
        ]
    )
    out = add_pchembl_columns(df)
    cols = list(out.columns)
    assert cols.index("pchembl_value_filled") < cols.index("b")
    assert cols.index("pchembl_relation") < cols.index("c")
