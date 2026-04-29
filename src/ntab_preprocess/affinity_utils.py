import numpy as np
import pandas as pd


_RELATION_INVERSION_MAP = {
    "=": "=",
    "~": "~",
    ">": "<",
    ">=": "<=",
    ">>": "<<",
    "<": ">",
    "<=": ">=",
    "<<": ">>",
}


def invert_relation(relation: str | None) -> str | None:
    """Invert an inequality relation symbol for pChEMBL direction.

    pChEMBL values are on a -log10 scale, which inverts inequality direction:
    e.g. IC50 > 1000 nM becomes pIC50 < 6.0.

    Args:
        relation: Relation string (e.g. '=', '>', '<=') or None.

    Returns:
        Inverted relation string, or None if input is None.

    Raises:
        ValueError: If relation is a non-None, non-NaN value not in the known map.
    """
    if relation is None:
        return None
    if isinstance(relation, float) and np.isnan(relation):
        return None
    if relation in _RELATION_INVERSION_MAP:
        return _RELATION_INVERSION_MAP[relation]
    raise ValueError(f"Unknown relation: {relation!r}")


def compute_pchembl_filled(row: pd.Series) -> float:
    """Compute pChEMBL value from standard_value when pchembl_value is missing.

    Uses standard_value (expected in nM) to compute -log10(standard_value * 1e-9).
    Returns NaN if units are not nM or standard_value is not numeric.

    Args:
        row: Series with columns 'standard_value' and 'standard_units'.

    Returns:
        Rounded pChEMBL value (2 decimal places), or NaN.
    """
    if row["standard_units"] != "nM":
        return np.nan
    try:
        val = float(row["standard_value"])
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(val):
        return np.nan
    return round(-np.log10(val * 1e-9), 2)


def add_pchembl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace pchembl_value and standard_relation with enriched versions.

    pchembl_value_filled replaces pchembl_value: pChEMBL value (-log10 scale)
        filled in for all rows, including censored measurements where
        standard_relation != '='. Uses ChEMBL's pchembl_value where available,
        falling back to computing it from standard_value (nM).

    pchembl_relation replaces standard_relation: The inequality relation in
        pChEMBL direction. Because -log10 inverts inequality direction,
        standard_relation is inverted (e.g. IC50 > x becomes pIC50 < y).

    The new columns are inserted at the same positions as the originals,
    which are dropped.

    Args:
        df: DataFrame with columns: pchembl_value, standard_value,
            standard_units, standard_relation.

    Returns:
        df with pchembl_value replaced by pchembl_value_filled and
        standard_relation replaced by pchembl_relation (does not modify in place).
    """
    result = df.copy()

    pchembl_pos = result.columns.get_loc("pchembl_value")
    result.insert(
        pchembl_pos,
        "pchembl_value_filled",
        result["pchembl_value"]
        .fillna(result.apply(compute_pchembl_filled, axis=1))
        .astype(float),
    )
    result = result.drop(columns=["pchembl_value"])

    relation_pos = result.columns.get_loc("standard_relation")
    result.insert(
        relation_pos,
        "pchembl_relation",
        result["standard_relation"].apply(invert_relation),
    )
    result = result.drop(columns=["standard_relation"])

    return result
