# Reproduce baseline model results on FEP+ 4 dataset

These steps reproduce the evaluation of the baseline model on the FEP+ 4 benchmark dataset. They assume that that you:
- have a local version of the ChEMBL database (v36) as a postres DB
- have downloaded or reproduced the FEP+ 4 datasplit files and benchmark data files from [here](https://github.com/bamattsson/paper-identifying_and_addressing_data_leakage/tree/627da788480c85266f83df5a8503465232325d95/data/out) and that they exist in `../paper-identifying_and_addressing_data_leakage/data/out/` relative to this repo.

If you want to rerun the predictions you can download the models from [this GDrive](https://drive.google.com/drive/folders/1HWiKaobRYdpQfH2dPTnXr1VFcCEIihWZ?usp=sharing) and only do steps 0 and 4.

If you want to retrain the models follow all the steps below. For FP + mol prop, FP only and mol prop only we trained the models on commit `52f3381`. The chemprop model on commit `ffb3e4e`.

**Step 0 – Install packages:**

```bash
uv sync --extra baseline
source .venv/bin/activate
```

**Step 1 – Run the data pipeline:**

This will run the full pipeline that pulls ChEMBL data and prepares the data files. First add your local ChEMBL (v36) credentials to `reproduce_results_on_fep4/01_dataset_generation.yaml`. Then run:

```bash
python src/ntab_preprocess/run_pipeline.py --config reproduce_results_on_fep4/01_dataset_generation.yaml
```

(Note that this will put everything in train split, we will override this split in the next step)

**Step 2 – Preprocess features and apply FEP+ split:**

```bash
python -m ntab_baseline.preprocess_training_data \
    --config reproduce_results_on_fep4/02_model_data.yaml
```

**Step 3 — Train:**

```bash
python -m ntab_baseline.train fit \
    --config reproduce_results_on_fep4/03_train.yaml
```

**Step 4 — Predict on the FEP+ benchmark:**

Replace `version_X` and `<best>` with your values and run:

```bash
python -m ntab_baseline.predict_on_csv \
    --checkpoint out_FEP4_baseline/lightning_logs/version_X/checkpoints/<best>.ckpt \
    --input-csv ../paper-identifying_and_addressing_data_leakage/data/out/FEPp_benchmark.csv \
    --size-weighted \
    --n-bootstraps 1000 \
    --output-csv predictions_version_X_FEPp4.csv
```

To predict on the OpenFE benchmark, use:

```bash
python -m ntab_baseline.predict_on_csv \
    --checkpoint out_FEP4_baseline/lightning_logs/version_X/checkpoints/<best>.ckpt \
    --input-csv ../paper-identifying_and_addressing_data_leakage/data/out/OpenFE_benchmark.csv \
    --extra-oov-mapping-file reproduce_results_on_fep4/openfe_oov_mapping.json \
    --size-weighted \
    --n-bootstraps 1000 \
    --min-assay-size 2 \
    --output-csv predictions_version_X_OpenFE.csv
```