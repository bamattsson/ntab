"""Shared constants for the binding prediction baseline."""

# ECFP4 fingerprint size (radius=2, 2048 bits).
FP_SIZE: int = 2048

# Minimum number of compounds an assay must have to contribute to the Pearson r metric.
MIN_ASSAY_SIZE: int = 10

# Mol property features fed into the model. Edit this list to include/exclude
# features — the model input dimension updates automatically.
MOL_PROP_FEATURES: list[str] = [
    "MolLogP",
    "ExactMolWt",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "FormalCharge",
    "MolMR",
    "FractionCSP3",
    "RingCount",
    "NumAromaticRings",
    "HeavyAtomCount",
]

N_MOL_PROP_FEATURES = len(MOL_PROP_FEATURES)

STANDARD_TYPE_INDEX: dict[str, int] = {"IC50": 0, "Ki": 1, "Kd": 2}
