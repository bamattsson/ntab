from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the tests/fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def chembl_smiles_sample() -> pd.DataFrame:
    """100-row sample from ChEMBL with columns: chembl_id, canonical_smiles, earliest_year."""
    return pd.read_csv(FIXTURES_DIR / "chembl_smiles_sample.csv")
