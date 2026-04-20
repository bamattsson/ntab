# Reproduce baseline model results on FEP+ 4 dataset

These steps reproduce the evaluation of the baseline model on the FEP+ 4 benchmark dataset. They assume that that you:
- have a local version of the ChEMBL database (v36) as a postres DB
- have downloaded or reproduced the FEP+ 4 datasplit files and benchmark data files from [here](https://github.com/bamattsson/paper-critical_assessment_of_binding_affinity_benchmarks/tree/421fc9a620c98e6e49517fcd2c253a3265ea4821/data/out) and that they exist in `../paper-critical_assessment_of_binding_affinity_benchmarks/data/out/` relative to this repo.

**Step 0 – Install packages:**

```bash
uv sync --extra model
source .venv/bin/activate
```

**Step 1 – Run the data pipeline:**

This will run the full pipeline that pulls ChEMBL data and prepares the data files. First add your local ChEMBL (v36) credentials to `reproduce_results_on_fep4/01_dataset_generation.yaml`. Then run:

```bash
python src/nfab/run_pipeline.py --config reproduce_results_on_fep4/01_dataset_generation.yaml
```

(Note that this will put everything in train split, we will override this split in the next step)

**Step 2 – Preprocess features and apply FEP+ split:**

```bash
python -m nfab_baseline.preprocess_training_data \
    --config reproduce_results_on_fep4/02_model_data.yaml
```

**Step 3 — Train:**

```bash
python -m nfab_baseline.train fit \
    --config reproduce_results_on_fep4/03_train.yaml
```

**Step 4 — Predict on the FEP+ benchmark:**

Replace `version_X` and `<best>` with your values and run:

```bash
python -m nfab_baseline.predict_on_csv \
    --checkpoint out_FEP4_baseline/lightning_logs/version_X/checkpoints/<best>.ckpt \
    --data-dir out_FEP4_baseline/data_preprocessing \
    --input-csv ../paper-critical_assessment_of_binding_affinity_benchmarks/data/out/FEPp_benchmark.csv \
    --output-csv predictions.csv \
    --size-weighted \
    --n-bootstraps 1000
```

---

On commit `52f3381` with a nvidia GPU we get the results 0.663, but the results can vary up or down a few 0.01 depending on random seeds.