import numpy as np
import pandas as pd
import pytest

from timesplit_affinity_benchmark.mol_fingerprints import compute_ecfp4_fingerprints


@pytest.fixture
def expected_fingerprints(fixtures_dir) -> tuple[np.ndarray, np.ndarray]:
    """Pre-computed ECFP4 fingerprints for the 100-row ChEMBL sample."""
    data = np.load(fixtures_dir / "ecfp4_expected.npz")
    return data["names"], data["fps"]


def test_compute_ecfp4_fingerprints(chembl_smiles_sample: pd.DataFrame) -> None:
    names, fps = compute_ecfp4_fingerprints(
        mol_names=chembl_smiles_sample["chembl_id"].tolist(),
        smiles=chembl_smiles_sample["canonical_smiles"].tolist(),
    )
    assert isinstance(names, np.ndarray)
    assert isinstance(fps, np.ndarray)
    assert len(names) == fps.shape[0]
    assert fps.shape[1] == 2048


def test_multiprocessing_matches_single(
    chembl_smiles_sample: pd.DataFrame,
    expected_fingerprints: tuple[np.ndarray, np.ndarray],
) -> None:
    """Multiprocessing output must be bit-identical to the pre-computed single-process result."""
    exp_names, exp_fps = expected_fingerprints

    names_mp, fps_mp = compute_ecfp4_fingerprints(
        mol_names=chembl_smiles_sample["chembl_id"].tolist(),
        smiles=chembl_smiles_sample["canonical_smiles"].tolist(),
        n_jobs=2,
    )

    assert list(names_mp) == list(exp_names)
    assert np.array_equal(fps_mp, exp_fps)
