"""Model utilities for the binding prediction baseline.

Contains:
- pearson_r_per_assay: the benchmark metric.
- MetricsPlotCallback: saves metrics_epoch.csv and metrics_plot.png after training.
- PredictWriterCallback: writes model predictions to a CSV file.
"""

import math

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from scipy.stats import pearsonr

from bind_pred_baseline.constants import MIN_ASSAY_SIZE


def pearson_r_per_assay(
    preds: np.ndarray,
    labels: np.ndarray,
    assay_ids: list[str],
    min_assay_size: int = MIN_ASSAY_SIZE,
    n_bootstrap: None | int = None,
) -> tuple[float, float | None]:
    """Compute mean Pearson r across assays, skipping assays below min_assay_size.

    Args:
        preds: Predicted values, shape (N,).
        labels: True values, shape (N,).
        assay_ids: Assay identifier per sample, length N.
        min_assay_size: Assays with fewer than this many samples are excluded.
        n_bootstrap: If set, also compute a bootstrap standard error using this
            many resamples (resampling at the assay level).

    Returns:
        Tuple of (pearson_r, se) where pearson_r is the size-weighted mean
        Pearson r across qualifying assays (NaN if none qualify), and se is
        the bootstrap standard error, or None if n_bootstrap was not provided.
    """
    assay_to_indices: dict[str, list[int]] = {}
    for i, assay in enumerate(assay_ids):
        assay_to_indices.setdefault(assay, []).append(i)

    rs: list[float] = []
    weights: list[int] = []
    for assay, indices in sorted(assay_to_indices.items()):
        if len(indices) < min_assay_size:
            continue
        idx = np.array(indices)
        r, _ = pearsonr(preds[idx], labels[idx])
        if not math.isfinite(r):
            continue
        rs.append(r)
        weights.append(len(indices))

    if not rs:
        return float("nan"), None

    rs_a = np.array(rs)
    w_a = np.array(weights, dtype=np.float64)
    pearson_r = float((rs_a * w_a).sum() / w_a.sum())
    if n_bootstrap is None:
        return pearson_r, None

    # Bootstrap a confidence interval by resampling assays
    rng = np.random.default_rng()
    num_assays = len(rs_a)
    boot_idx = rng.integers(0, num_assays, size=(n_bootstrap, num_assays))
    boot_rs = rs_a[boot_idx]  # (n_bootstrap, num_assays)
    boot_ws = w_a[boot_idx]   # (n_bootstrap, num_assays)
    boot_means = (boot_rs * boot_ws).sum(axis=1) / boot_ws.sum(axis=1)
    se = float(boot_means.std())
    return pearson_r, se


class MetricsPlotCallback(L.Callback):
    """Saves metrics_epoch.csv and metrics_plot.png at the end of training."""

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        try:
            import csv
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            metrics_csv = f"{trainer.logger.log_dir}/metrics.csv"
            epochs_val, val_loss, val_pearson_r, train_epochs, train_loss = [], [], [], [], []
            epoch_rows, all_fieldnames = [], []
            with open(metrics_csv) as f:
                reader = csv.DictReader(f)
                all_fieldnames = reader.fieldnames or []
                for row in reader:
                    epoch = int(row["epoch"])
                    is_epoch_row = bool(
                        row.get("val_loss") or row.get("val_pearson_r") or row.get("train_loss_epoch")
                    )
                    if is_epoch_row:
                        epoch_rows.append(row)
                    if row.get("val_loss") and row.get("val_pearson_r"):
                        epochs_val.append(epoch)
                        val_loss.append(float(row["val_loss"]))
                        val_pearson_r.append(float(row["val_pearson_r"]))
                    if row.get("train_loss_epoch"):
                        train_epochs.append(epoch)
                        train_loss.append(float(row["train_loss_epoch"]))

            metrics_epoch_csv = f"{trainer.logger.log_dir}/metrics_epoch.csv"
            with open(metrics_epoch_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                writer.writeheader()
                writer.writerows(epoch_rows)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].plot(train_epochs, train_loss, "b-o", ms=4)
            axes[0].set(title="Train Loss (MSE)", xlabel="Epoch", ylabel="MSE")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(epochs_val, val_loss, "r-o", ms=4)
            axes[1].set(title="Val Loss (MSE)", xlabel="Epoch", ylabel="MSE")
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(epochs_val, val_pearson_r, "g-o", ms=4)
            axes[2].set(title="Val Pearson r (size-weighted)", xlabel="Epoch", ylabel="Pearson r")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = f"{trainer.logger.log_dir}/metrics_plot.png"
            plt.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"plot saved → {plot_path}")
        except Exception:
            import traceback
            traceback.print_exc()


class PredictWriterCallback(BasePredictionWriter):
    """Writes model predictions to a CSV file at the end of prediction.

    Args:
        output_csv: Path to the output CSV file.
    """

    def __init__(self, output_csv: str) -> None:
        super().__init__(write_interval="epoch")
        self.output_csv = output_csv

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx) -> None:
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices) -> None:
        """Flatten all batches and write to CSV.

        Args:
            predictions: list[dict] — one dict per batch, as returned by predict_step.
        """
        rows = []
        for batch_dict in predictions:
            names = batch_dict["ligand_name"]
            uniprot_ids = batch_dict["uniprot_id"]
            pred_pchembl = batch_dict["pred_pchembl"]
            for name, uid, pred in zip(names, uniprot_ids, pred_pchembl):
                rows.append({
                    "ligand_name": name,
                    "uniprot_id": uid,
                    "pred_pchembl": float(pred),
                })
        df = pd.DataFrame(rows, columns=["ligand_name", "uniprot_id", "pred_pchembl"])
        df.to_csv(self.output_csv, index=False)
        print(f"Predictions saved to {self.output_csv}")
