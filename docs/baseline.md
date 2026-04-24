# Baseline model: `nfab_baseline`

An MLP baseline for protein-ligand binding affinity prediction, trained on
ChEMBL data using ECFP4 fingerprints and physicochemical descriptors.

---

## Model architecture

```
Inputs: ECFP4 fingerprints (2048 bits) + 12 mol properties + target embedding
  fp_encoder:       Linear(2048, hidden) → BatchNorm → GELU
  target_embedding: Embedding(n_targets, 256)
  concatenate:      [fp_encoded | mol_props | target_embed]
  head (×2):        Linear(hidden) → BatchNorm → GELU
                    Linear(hidden, 1)
  output:           head(combined) + target_bias + assay_type_bias
```

Assay type biases are learned separately for IC50, Ki, and Kd.
The primary validation metric is size-weighted mean Pearson r across assays
(minimum 10 compounds per assay).

---

## Setup

Install with the `[model]` extras (includes torch, lightning, biopython, etc.):

```bash
uv sync --extra baseline
source .venv/bin/activate
```

---

## Usage

### Step 1 — Preprocess training data

Requires `activities.parquet`, and `targets.parquet`
from the benchmark pipeline (see main README).

```bash
python -m nfab_baseline.preprocess_training_data \
    --config configs/baseline/data.yaml
```

Writes preprocessed data to `out/data_preprocessing/`:

| File | Contents |
|---|---|
| `fingerprints_ecfp4.npz` | ECFP4 count fingerprints, shape (N_compounds, 2048) |
| `mol_properties.npz` | 12 normalised physicochemical properties + scaler |
| `train.npz`, `val.npz`, `test.npz` | Split indices, labels, assay IDs (val/test combine all matching `val_sim_*`/`test_sim_*` bins) |
| `target_index.json` | `uniprot_id → integer index` mapping |
| `meta.json` | `n_targets`, `n_standard_types`, `fp_size`, `fp_type` |
| `oov_target_mapping.json` | OOV target → nearest training target (if any OOV exist) |

OOV targets (present in val/test but not train) are automatically mapped to
the most sequence-similar training target via BLOSUM62 global alignment.

### Step 2 — Train

```bash
python -m nfab_baseline.train fit --config configs/baseline/train.yaml
```

The best checkpoint (by `val_pearson_r`) is saved under
`out/lightning_logs/version_X/checkpoints/`.

### Step 3 — Evaluate on benchmark splits

Run inference on one or more held-out splits from `activities.parquet` and
write per-row predictions to a CSV:

```bash
python -m nfab_baseline.predict_on_benchmark \
    --checkpoint out_baseline/lightning_logs/version_X/checkpoints/<best>.ckpt \
    --data-dir out_baseline/data_preprocessing \
    --activities out/activities.parquet \
    --targets out/targets.parquet \
    --splits test_sim_0.00_0.35 test_sim_0.35_0.50 test_sim_0.50_0.70 test_sim_0.70_1.00 test_sim_1.00 \
    --output predictions.csv
```

Prints size-weighted Pearson r per split (and overall when multiple splits are
requested). The output CSV has columns: `assay_id`, `ligand_name`,
`uniprot_id`, `standard_type`, `split`, `pchembl_value`, `pred_pchembl`.

### Step 4 — Predict on new compounds

Provide a CSV with columns `ligand_name`, `uniprot_id`, `smiles` (and
optionally `standard_type` and `pchembl_value`):

```bash
python -m nfab_baseline.predict_on_csv \
    --checkpoint out_baseline/lightning_logs/version_X/checkpoints/<best>.ckpt \
    --data-dir out_baseline/data_preprocessing \
    --input-csv compounds.csv \
    --output-csv predictions.csv
```

---

## Mol property features

The 12 physicochemical descriptors computed by RDKit:

`MolLogP`, `ExactMolWt`, `TPSA`, `NumHDonors`, `NumHAcceptors`,
`NumRotatableBonds`, `FormalCharge`, `MolMR`, `FractionCSP3`,
`RingCount`, `NumAromaticRings`, `HeavyAtomCount`

Properties are normalised (mean/std) using training-set statistics only.
