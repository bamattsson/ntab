# Novelty-Tiered Affinity Benchmark (NTAB)

**Paper**: [coming soon]() | **Dataset**: [coming soon]()

A novelty-tiered, time-split benchmark for protein-ligand binding affinity prediction, built from ChEMBL binding data.

## Overview

The benchmark is constructed from ChEMBL binding affinity data using two complementary strategies to prevent data leakage:

1. **Time split**: activities are partitioned by assay publication year (train: before 2022, val: 2022, test: 2023+).
2. **Similarity-binned test/val sets**: all test and val compounds are labelled by their maximum Morgan Fingerprint (radius 2, 2048-bit) Tanimoto similarity to pre-cutoff compounds, yielding five bins: `[0, 0.35)`, `[0.35, 0.5)`, `[0.5, 0.7)`, `[0.7, 1.0)`, and `=1.0`.

Val and test assays are further filtered to retain only well-characterised (assay, measurement-type) groups: minimum 10 unique compounds, pChEMBL SD ≥ 0.5, equality-relation measurements only, and at most one assay per publication.

## Evaluating your model

### 1. Download the dataset

Download `activities.parquet` and `targets.parquet` from [coming soon](). The download also includes predictions from the baseline models, which you can use as a reference when comparing your model's performance.

### 2. Train your model

Use all rows where `split = train` for train, and where `split = val_*` for validation. See [docs/benchmark_guidelines.md](docs/benchmark_guidelines.md) for more details on how we suggest people use this benchmark.

### 3. Generate predictions

Generate a prediction for every row where `split` starts with `test_`. Save these as a CSV with the following columns:

| Column | Description |
|---|---|
| `assay_id` | ChEMBL assay ID |
| `ligand_name` | ChEMBL compound ID |
| `standard_type` | Measurement type (`Ki`, `Kd`, or `IC50`) |
| `pred_pchembl` | Predicted pChEMBL value |

Rows not present in the CSV are filled with `pred_pchembl = 6.0` (1 µM) when computing metrics.


### 4. Install dependencies for analysis

#### uv (recommended)

[uv](https://github.com/astral-sh/uv) is recommended. Requires Python 3.10+.

```bash
uv sync
source .venv/bin/activate
```

#### conda

```bash
conda create -n ntab python=3.11
conda activate ntab
pip install -e .
```

### 5. Analyse results

Open `calculate_benchmark_performance.ipynb`, set `ACTIVITIES_PATH` and `MODELS`, and run all cells. The notebook computes mean Pearson r per similarity bin with bootstrap confidence intervals and produces a summary plot.

## Dataset

### `activities.parquet`

One row per activity measurement. Columns:

| Column | Description |
|---|---|
| `target_chembl_id` | ChEMBL target ID |
| `assay_chembl_id` | ChEMBL assay ID |
| `ligand_chembl_id` | ChEMBL compound ID |
| `standard_type` | Measurement type: `Ki`, `Kd`, or `IC50` |
| `pchembl_relation` | Inequality relation on the pChEMBL scale (e.g. `<`, `=`). Inverted from `standard_relation` because −log10 reverses inequality direction. |
| `pchembl_value_filled` | pChEMBL value (−log10 scale). Uses ChEMBL's `pchembl_value` where available; otherwise computed. |
| `split` | Split label (see below) |
| `mw_freebase` | Molecular weight of the parent compound (salt-stripped), from ChEMBL `compound_properties` |
| `data_validity_comment` | ChEMBL data validity flag (`NULL` or `Manually validated`) |
| `potential_duplicate` | ChEMBL duplicate flag |
| `doc_year` | Publication year of the assay document |
| `cpd_earliest_year` | Earliest publication year for the compound across all ChEMBL records |
| `max_sim_pre_2023` | Max Tanimoto similarity (ECFP4) to any compound with `cpd_earliest_year < 2023` (test novelty cutoff) |
| `most_sim_cpd_pre_2023` | ChEMBL ID of the most similar pre-2023 compound |
| `max_sim_pre_2022` | Max Tanimoto similarity (ECFP4) to any compound with `cpd_earliest_year < 2022` (val novelty cutoff) |
| `most_sim_cpd_pre_2022` | ChEMBL ID of the most similar pre-2022 compound |
| `canonical_smiles` | Canonical SMILES from ChEMBL `compound_structures` |

#### Split labels

Test and val rows are labelled by which similarity bin their compound falls into. With the default bins in `configs/benchmark.yaml`:

| Label | Condition |
|---|---|
| `train` | `doc_year < 2022` |
| `test_sim_0.00_0.35` | `doc_year >= 2023` and `max_sim_pre_2023 ∈ [0.00, 0.35)` |
| `test_sim_0.35_0.50` | `doc_year >= 2023` and `max_sim_pre_2023 ∈ [0.35, 0.50)` |
| `test_sim_0.50_0.70` | `doc_year >= 2023` and `max_sim_pre_2023 ∈ [0.50, 0.70)` |
| `test_sim_0.70_1.00` | `doc_year >= 2023` and `max_sim_pre_2023 ∈ [0.70, 1.00)` |
| `test_sim_1.00` | `doc_year >= 2023` and `max_sim_pre_2023 = 1.00` (identical to a pre-2023 compound) |
| `val_sim_*` | same pattern, `doc_year == 2022`, using `max_sim_pre_2022` |

Rows with no `doc_year` are excluded from all splits. Rows whose similarity does not fall into any configured bin are also excluded. Bins are configurable via `test_set_similarity_bins` in the pipeline config.

### `targets.parquet`

One row per single-protein target. Columns: `target_chembl_id`, `uniprot_id`, `gene_name`, `target_class`, `target_family`, `organism`, `target_name`, `sequence`.

### Data inclusion criteria

The following filters are applied when querying ChEMBL:

| Criterion | Value | Rationale |
|---|---|---|
| Target type | `SINGLE PROTEIN` only | Excludes protein complexes, cell lines, organisms, etc. |
| Assay type | `B` (binding) only | Excludes functional, ADMET, and other non-binding assays |
| Confidence score | `9` (maximum) | Direct single-protein target assignment; excludes lower-confidence mappings |
| Relationship type | `D` or `H` | Direct or homologous target mappings; excludes non-specific and unknown relationships |
| Measurement type | `Ki`, `Kd`, or `IC50` | Standard binding affinity readouts |
| Data validity | `NULL` or `Manually validated` | Excludes flagged/unreliable entries |
| Protein sequence | Wildtype only | Excludes assays against mutant sequences (annotated in `variant_sequences`) |

After ChEMBL retrieval, three further filters determine which rows appear in the final output:

- **No `doc_year`**: rows with a missing publication year cannot be assigned to a split and are excluded.
- **Similarity bin (test/val only)**: rows whose compound similarity does not fall into any configured bin are excluded. All compounds are retained when the bins cover the full [0, 1.0] range.
- **Assay quality (test/val only)**: (assay, measurement-type) groups are removed if they have fewer than 10 unique compounds, a pChEMBL SD below 0.5, non-equality relations, or share a publication with another passing assay. Configurable via `filter_val_and_test_sets` in the config.

## Reproducing the benchmark

To regenerate the dataset from ChEMBL, see [docs/preprocess.md](docs/preprocess.md).

## Baseline model

A ligand-only MLP baseline is included as a probe for what performance is achievable through pure ligand memorisation. For training and inference instructions see [docs/baseline.md](docs/baseline.md).

## Citation

If you use NTAB in your research, please cite:

```bibtex
@article{mattsson2026identifying,
  title   = {Identifying and Addressing Systematic Data Leakage in Protein-Ligand Affinity Benchmarks},
  author  = {Mattsson, Bj{\"o}rn and Walters, W. Patrick},
  year = {2026},
  doi = {},
  journal = {},
}
```

## Development

```bash
uv sync --extra baseline --extra preprocess --extra dev
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```
