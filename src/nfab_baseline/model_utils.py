"""Model utilities for the binding prediction baseline.

Contains:
- MetricsPlotCallback: saves metrics_epoch.csv and metrics_plot.png after training.
"""

import lightning as L


class MetricsPlotCallback(L.Callback):
    """Saves metrics_epoch.csv and metrics_plot.png at the end of training."""

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        try:
            import csv
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            metrics_csv = f"{trainer.logger.log_dir}/metrics.csv"
            epochs_val, val_loss, val_pearson_r, train_epochs, train_loss = (
                [],
                [],
                [],
                [],
                [],
            )
            epoch_rows, all_fieldnames = [], []
            with open(metrics_csv) as f:
                reader = csv.DictReader(f)
                all_fieldnames = reader.fieldnames or []
                for row in reader:
                    epoch = int(row["epoch"])
                    is_epoch_row = bool(
                        row.get("val_loss")
                        or row.get("val_pearson_r")
                        or row.get("train_loss_epoch")
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
            axes[2].set(
                title="Val Pearson r (macro avg)",
                xlabel="Epoch",
                ylabel="Pearson r",
            )
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = f"{trainer.logger.log_dir}/metrics_plot.png"
            plt.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"plot saved → {plot_path}")
        except Exception:
            import traceback

            traceback.print_exc()
