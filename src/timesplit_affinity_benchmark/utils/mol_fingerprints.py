import multiprocessing
from functools import partial

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from tqdm import tqdm


def _process_one(args: tuple[str, str]) -> tuple[str, np.ndarray] | None:
    """Parse one SMILES and return (name, fingerprint) or None on failure.

    Must be a module-level function so it can be pickled for multiprocessing.
    """
    name, smi = args
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    if mol is None:
        print(f"WARNING: could not parse SMILES, skipping. name={name}, smiles={smi}")
        return None
    mol = AllChem.RemoveAllHs(mol)
    return name, fpgen.GetFingerprintAsNumPy(mol)


def compute_ecfp4_fingerprints(
    mol_names: list[str],
    smiles: list[str],
    n_jobs: int = 1,
    chunksize: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ECFP4 fingerprints (radius=2, size=2048) for a list of molecules.

    Molecules that fail to parse are skipped and excluded from the output.

    Preprocessing applied:
    - Sanitization via RDKit (aromaticity, valence checking)
    - Explicit Hs removed
    - Note: multi-component SMILES (e.g. salts like "CC(=O)[O-].[Na+]") are NOT
      stripped to the largest fragment. ~4% of ChEMBL SMILES are affected. This is
      a conscious choice to keep the preprocessing simple; be aware that counterion
      fragments will influence the fingerprint for those molecules.

    Args:
        mol_names: Molecule identifiers, one per SMILES.
        smiles: SMILES strings.
        n_jobs: Number of worker processes. 1 = single-process (default).
            -1 = use all available CPUs (``multiprocessing.cpu_count()``).
        chunksize: Number of molecules sent to each worker at a time.
            Larger values reduce IPC overhead but increase memory per worker.

    Returns:
        Tuple of (names, fingerprints) as numpy arrays, with failed parses removed.
        names shape: (N,), fingerprints shape: (N, 2048).
    """
    pairs = list(zip(mol_names, smiles))

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs == 1:
        results = [_process_one(p) for p in tqdm(pairs, desc="Computing ECFP4")]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_one, pairs, chunksize=chunksize),
                    total=len(pairs),
                    desc=f"Computing ECFP4 ({n_jobs} workers)",
                )
            )

    out_names = []
    out_fps = []
    for r in results:
        if r is not None:
            out_names.append(r[0])
            out_fps.append(r[1])

    return np.array(out_names), np.array(out_fps)
