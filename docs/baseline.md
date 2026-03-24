# Baseline model: `bind_pred_baseline`

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
uv pip install -e ".[model]"
```

---

## Usage

### Step 1 — Preprocess training data

Requires `activities.parquet`, `targets.parquet`, and `compounds_raw.parquet`
from the benchmark pipeline (see main README).

```bash
uv run python -m bind_pred_baseline.preprocess_training_data \
    --config configs/baseline/data.yaml
```

Writes preprocessed data to `out/data_preprocessing/`:

| File | Contents |
|---|---|
| `fingerprints_ecfp4.npz` | ECFP4 count fingerprints, shape (N_compounds, 2048) |
| `mol_properties.npz` | 12 normalised physicochemical properties + scaler |
| `train.npz`, `val.npz`, `test.npz` | Split indices, labels, assay IDs |
| `target_index.json` | `uniprot_id → integer index` mapping |
| `meta.json` | `n_targets`, `n_standard_types`, `fp_size`, `fp_type` |
| `oov_target_mapping.json` | OOV target → nearest training target (if any OOV exist) |

OOV targets (present in val/test but not train) are automatically mapped to
the most sequence-similar training target via BLOSUM62 global alignment.

### Step 2 — Train

```bash
uv run python -m bind_pred_baseline.train fit --config configs/baseline/train.yaml
```

The best checkpoint (by `val_pearson_r`) is saved under
`out/lightning_logs/version_X/checkpoints/`.

### Step 3 — Evaluate on test set

```bash
uv run python -m bind_pred_baseline.train test \
  --config out/lightning_logs/version_X/config.yaml \
  --ckpt_path out/lightning_logs/version_X/checkpoints/<best>.ckpt
```

### Step 4 — Predict on new compounds

Edit `configs/baseline/predict.yaml` to set `predict_input_csv` (CSV with
columns `ligand_name`, `uniprot_id`, `smiles`), then run:

```bash
uv run python -m bind_pred_baseline.train predict \
  --config out/lightning_logs/version_X/config.yaml \
  --ckpt_path out/lightning_logs/version_X/checkpoints/<best>.ckpt \
  --config configs/baseline/predict.yaml
```

Predictions are written to `predictions.csv`.

---

## Reproduce FEP+ benchmark results

These steps reproduce the evaluation on the FEP+ benchmark dataset. They
assume that the FEP+ data files are available at
`../paper_data_leakage_FEPp_benchmark/data/out/`.

**Step 1 — Preprocess with the FEP+ data split:**

```bash
uv run python -m bind_pred_baseline.preprocess_training_data \
    --config configs/baseline_fep_split/data.yaml
```

**Step 2 — Train:**

```bash
uv run python -m bind_pred_baseline.train fit \
    --config configs/baseline_fep_split/train.yaml
```

**Step 3 — Predict on the FEP+ benchmark:**

```bash
uv run python -m bind_pred_baseline.train predict \
  --config out/lightning_logs/version_X/config.yaml \
  --ckpt_path out/lightning_logs/version_X/checkpoints/<best>.ckpt \
  --config configs/baseline_fep_split/predict.yaml
```

**Step 4 — Evaluate Pearson r:**

```bash
uv run python scripts/eval_fep_benchmark.py \
  --predictions predictions.csv \
  --benchmark ../paper_data_leakage_FEPp_benchmark/data/out/FEPp_benchmark.csv
```

Per-assay Pearson r and the overall size-weighted mean are printed to stdout.

---

## Mol property features

The 12 physicochemical descriptors computed by RDKit:

`MolLogP`, `ExactMolWt`, `TPSA`, `NumHDonors`, `NumHAcceptors`,
`NumRotatableBonds`, `FormalCharge`, `MolMR`, `FractionCSP3`,
`RingCount`, `NumAromaticRings`, `HeavyAtomCount`

Properties are normalised (mean/std) using training-set statistics only.
The scaler is stored in `mol_properties.npz` and reused at inference time.
