# timesplit-affinity-benchmark

Code for generating a time-split, novelty-filtered benchmark for protein-ligand binding affinity prediction.

## Overview

The benchmark is constructed from ChEMBL binding affinity data using two complementary strategies to prevent data leakage:

1. **Time split**: activities are partitioned by assay publication year (train: before 2023, val: 2023, test: 2024+).
2. **Novelty filter**: test and validation compounds are filtered by ECFP4 Tanimoto similarity — a compound is kept only if its maximum similarity to any earlier compound is below 0.35.

## Installation

### uv (recommended)

[uv](https://github.com/astral-sh/uv) is recommended. Requires Python 3.10+.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### conda

```bash
conda create -n timesplit-affinity python=3.10
conda activate timesplit-affinity
pip install -e .
```

## Reproducing the data

### 1. Set up ChEMBL

The benchmark is generated from a local ChEMBL PostgreSQL database. We used ChEMBL 36, but other versions should work as well. To set it up:

1. Access the psql interface and create a new database:
   ```bash
   sudo -u postgres psql
   create database chembl_36;
   ```

2. Download the `chembl_36_postgresql.tar.gz` file and extract it:
   ```bash
   tar xvzf chembl_36_postgresql.tar.gz
   ```

3. Restore the database (run from bash, not from inside psql):
   ```bash
   pg_restore --no-owner -U <user> --dbname=chembl_36 chembl_36/chembl_36_postgresql/chembl_36_postgresql.dmp --verbose
   ```

### 2. Configure the pipeline

Add your database credentials to `config.yaml` and adjust `n_jobs` to match your CPU count.

### 3. Run the pipeline

```bash
python src/timesplit_affinity_benchmark/run_pipeline.py --config config.yaml
```

The pipeline runs in 7 steps and writes intermediate files to `intermediate_out/` and final outputs to `out/`. The full ChEMBL 36 run takes approximately 20 minutes on a machine with 32 cores and requires ~40 GB of RAM, dominated by loading fingerprints for the Tanimoto similarity computation.

## Output files

### `out/activities.parquet`

One row per activity measurement. Columns:

| Column | Description |
|---|---|
| `target_chembl_id` | ChEMBL target ID |
| `assay_chembl_id` | ChEMBL assay ID |
| `ligand_chembl_id` | ChEMBL compound ID |
| `standard_type` | Measurement type: `Ki`, `Kd`, or `IC50` |
| `pchembl_relation` | Inequality relation on the pChEMBL scale (e.g. `<`, `=`). Inverted from `standard_relation` because −log10 reverses inequality direction. |
| `pchembl_value_filled` | pChEMBL value (−log10 scale). Uses ChEMBL's `pchembl_value` where available; otherwise computed as −log10(`standard_value` × 10⁻⁹) for nM measurements. |
| `split` | Split label (see below) |
| `mw_freebase` | Molecular weight of the parent compound (salt-stripped), from ChEMBL `compound_properties` |
| `mutation` | Variant/mutation annotation from ChEMBL `variant_sequences`, if available |
| `data_validity_comment` | ChEMBL data validity flag (`NULL` or `Manually validated`) |
| `potential_duplicate` | ChEMBL duplicate flag |
| `doc_year` | Publication year of the assay document |
| `cpd_earliest_year` | Earliest publication year for the compound across all ChEMBL records |
| `max_sim_pre_2024` | Max Tanimoto similarity (ECFP4) to any compound with `cpd_earliest_year < 2024` |
| `most_sim_cpd_pre_2024` | ChEMBL ID of the most similar pre-2024 compound |
| `max_sim_pre_2023` | Max Tanimoto similarity (ECFP4) to any compound with `cpd_earliest_year < 2023` |
| `most_sim_cpd_pre_2023` | ChEMBL ID of the most similar pre-2023 compound |
| `canonical_smiles` | Canonical SMILES from ChEMBL `compound_structures` |

#### Split labels

| Label | Condition |
|---|---|
| `train` | `doc_year < 2023` |
| `val_novel` | `doc_year == 2023` and compound is novel vs. pre-2023 compounds |
| `val_not_novel` | `doc_year == 2023` and compound is not novel vs. pre-2023 compounds |
| `test` | `doc_year >= 2024` and compound is novel vs. pre-2024 compounds |
| `2024_not_novel` | `doc_year >= 2024` and compound is not novel (excluded by default) |

A compound is **novel** if its maximum ECFP4 Tanimoto similarity to all earlier compounds is strictly below `tanimoto_threshold` (default 0.35). Rows with no `doc_year` are excluded from all splits.

### `out/targets.parquet`

One row per single-protein target. Columns: `target_chembl_id`, `uniprot_id`, `gene_name`, `target_class`, `target_family`, `organism`, `target_name`, `sequence`.

### `intermediate_out/`

Intermediate files written between steps for inspection and debugging:

| File | Contents |
|---|---|
| `activities_raw.parquet` | Raw activity query results from ChEMBL |
| `compounds_raw.parquet` | Compounds (SMILES, earliest year, MW) filtered to the active activities |
| `targets_raw.parquet` | Single-protein targets with sequence and classification |
| `assay_docs.parquet` | Document metadata per assay: `assay_chembl_id`, `doc_chembl_id`, `doi`, `title`, `src_description` |
| `fingerprints.npz` | ECFP4 fingerprint matrix (`fps`) and compound IDs (`names`) |
| `compounds_with_novelty.parquet` | Compounds enriched with novelty columns for both the 2023 and 2024 cutoffs |
| `split_assignments.parquet` | Activities with split labels before final column selection and filtering |

## Running tests

```bash
uv pip install -e ".[dev]"
pytest
```
