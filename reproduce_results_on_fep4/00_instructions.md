# Reproduce baseline model results on FEP+ 4 dataset

These steps reproduce the evaluation of the baseline model on the FEP+ 4 benchmark dataset. They assume that that you:
- have a local version of the ChEMBL database (v36) as a postres DB
- have downloaded or reproduced the FEP+ 4 datasplit files and benchmark data files from [here](https://github.com/bamattsson/paper_data_leakage_FEPp_benchmark/tree/4c9cd86fd2891cfef11769a2a4b19bc92fb3b5e3/data/out) and that they exist in `../paper_data_leakage_FEPp_benchmark/data/out/` relative to this repo.

**Step 0 – Install packages:**

```bash
uv sync --extra model
source .venv/bin/activate
```

**Step 1 – Run the data pipeline:**

This will run the full pipeline that pulls ChEMBL data and prepares the data files. First add your local ChEMBL (v36) credentials to `reproduce_results_on_fep4/01_dataset_generation.yaml`. Then run:

```bash
python src/timesplit_affinity_benchmark/run_pipeline.py --config reproduce_results_on_fep4/01_dataset_generation.yaml
```

(Note that this will apply a time-based split as per the design idea of this repo, we will override this in the next step)

**Step 2 – Preprocess features and apply FEP+ split:**

```bash
python -m bind_pred_baseline.preprocess_training_data \
    --config reproduce_results_on_fep4/02_model_data.yaml
```

**Step 3 — Train:**

```bash
python -m bind_pred_baseline.train fit \
    --config reproduce_results_on_fep4/03_train.yaml
```

**Step 4 — Predict on the FEP+ benchmark:**

Replace `version_X` and `<best>` with your values and run:

```bash
python -m bind_pred_baseline.train predict \
  --config out_baseline/lightning_logs/version_X/config.yaml \
  --ckpt_path out_baseline/lightning_logs/version_X/checkpoints/<best>.ckpt \
  --config reproduce_results_on_fep4/04_predict.yaml
```

**Step 5 — Evaluate Pearson r:**

```bash
python reproduce_results_on_fep4/eval_fep_benchmark.py \
  --predictions predictions.csv \
  --benchmark ../paper_data_leakage_FEPp_benchmark/data/out/FEPp_benchmark.csv
```

Per-assay Pearson r and the overall size-weighted mean are printed to stdout.

---

On commit `2690a81c4` with a nvidia GPU we get the results 0.663, but the results can vary up or down a few 0.01 depending on random seeds.