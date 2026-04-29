# Reproducing the benchmark: `ntab_preprocess`

Instructions for regenerating the NTAB dataset from a local ChEMBL database.

---

## Installation

```bash
uv sync --extra preprocess
source .venv/bin/activate
```

## Prerequisites: Set up ChEMBL

The benchmark is generated from a local ChEMBL PostgreSQL database. See [docs/chembl_setup.md](chembl_setup.md) for instructions on setting up this database locally.

## 1. Configure the pipeline

Add your ChEMBL database credentials to `configs/benchmark.yaml`.

## 2. Run the pipeline

```bash
python -m ntab_preprocess.run_pipeline --config configs/benchmark.yaml
```

The pipeline runs in 8 steps and writes intermediate files to `out/intermediate/` and final outputs to `out/`. The full ChEMBL 36 run takes approximately 45 minutes on a machine with 16 cores and requires ~40 GB of RAM.

## Intermediate output files

| File | Contents |
|---|---|
| `activities_raw.parquet` | Raw activity query results from ChEMBL |
| `compounds_raw.parquet` | Compounds (SMILES, earliest year, MW) filtered to the active activities |
| `targets_raw.parquet` | Single-protein targets with sequence and classification |
| `assay_docs.parquet` | Document metadata per assay: `assay_chembl_id`, `doc_chembl_id`, `doi`, `title`, `src_description` |
| `fingerprints.npz` | ECFP4 fingerprint matrix (`fps`) and compound IDs (`names`) |
| `compounds_with_novelty.parquet` | Compounds enriched with similarity columns (`max_sim_pre_*`, `most_sim_cpd_pre_*`) for both the 2022 and 2023 cutoffs |
| `split_assignments.parquet` | Activities with split labels before final column selection and filtering |
