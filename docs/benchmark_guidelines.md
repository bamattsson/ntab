# Benchmark Guidelines

## Training data

- Use the **train** and **val** splits in whatever way you like — mix them, re-split, filter.
- Filtering training rows is fine and may even help (e.g. removing inequality-relation measurements like `<` and `>`).
- **Do not train on compounds outside the provided train and val splits.** The similarity bins measure novelty relative to the compounds in the training split specifically — not all of ChEMBL and not external datasets. Adding extra compounds (from external sources or other ChEMBL assay types) means your model may have seen structurally similar compounds that were absent from the reference set, invalidating the novelty comparison. If you need extra training data, you can regenerate the benchmark with those compounds included in the reference set — see [preprocess.md](preprocess.md) — but your results will not be directly comparable to others.
- Do not use pre-trained ML models, as they may have been trained on different training data than what is allowed here.

## Test set

- Use the **val set**, not test, for model selection and hyperparameter tuning.
- **Do not re-split the test set** into test and val to tune hyperparameters — leakage can happen across val and test boundaries as well.

## Sharing predictions

To make it easy for others to include your results in comparisons, please make your prediction file publicly available and link to it in your publication. The format is defined in the [README](../README.md) under *Generate predictions*. We may move to HuggingFace, Polaris, or a similar platform in the future to standardise submission and comparison.
