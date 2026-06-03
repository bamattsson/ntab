"""Lightning callbacks and artifact utilities for the baseline model."""

from __future__ import annotations

import shutil
from pathlib import Path

import lightning as L


def resolve_data_dir(checkpoint_path: Path) -> Path:
    """Find the training artifacts directory from a checkpoint path.

    Looks for ``target_index.json`` in the checkpoint's grandparent directory
    (the Lightning version directory, e.g. ``version_0/``).

    Args:
        checkpoint_path: Path to the ``.ckpt`` file.

    Returns:
        Path to the directory containing training artifacts.

    Raises:
        FileNotFoundError: If the required artifact files are not found.
    """
    # checkpoints/ sits one level below the version dir
    version_dir = checkpoint_path.resolve().parent.parent
    marker = version_dir / "target_index.json"
    if not marker.exists():
        raise FileNotFoundError(
            f"Could not find target_index.json next to the checkpoint. "
            f"Looked in {version_dir}."
        )
    return version_dir


class CopyTrainingArtifacts(L.Callback):
    """Copy training artifacts (target_index.json, meta.json) into the Lightning log directory.

    This ensures prediction scripts can find these files alongside the checkpoint
    without needing a separate --data-dir argument.

    Reads ``data_dir`` from ``trainer.datamodule.data_dir`` at train start.
    """

    REQUIRED = ("target_index.json", "meta.json")
    OPTIONAL = ("oov_target_mapping.json",)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.data_dir = Path(trainer.datamodule.data_dir)
        log_dir = Path(trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        for name in self.REQUIRED:
            src = self.data_dir / name
            if not src.exists():
                raise FileNotFoundError(f"Required artifact not found: {src}")
            dst = log_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)

        for name in self.OPTIONAL:
            src = self.data_dir / name
            dst = log_dir / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
