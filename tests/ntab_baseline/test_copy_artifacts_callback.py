"""Tests for the CopyTrainingArtifacts callback."""

import json
from pathlib import Path

import lightning as L
import pytest
import torch

from ntab_baseline.callbacks import CopyTrainingArtifacts


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a fake data_dir with target_index.json and meta.json."""
    d = tmp_path / "data_preprocessing"
    d.mkdir()
    (d / "target_index.json").write_text(json.dumps({"P12345": 0, "Q67890": 1}))
    (d / "meta.json").write_text(
        json.dumps({"n_targets": 2, "fp_size": 1024, "fp_type": "ecfp4"})
    )
    return d


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Create a fake Lightning log directory."""
    d = tmp_path / "lightning_logs" / "version_0"
    d.mkdir(parents=True)
    return d


class FakeDataModule:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = str(data_dir)


class FakeTrainer:
    """Minimal stand-in for a Lightning Trainer."""

    def __init__(self, log_dir: Path, data_dir: Path) -> None:
        self.log_dir = str(log_dir)
        self.datamodule = FakeDataModule(data_dir)


class TestCopyTrainingArtifacts:
    def test_copies_files_on_train_start(
        self, data_dir: Path, log_dir: Path
    ) -> None:
        cb = CopyTrainingArtifacts()
        trainer = FakeTrainer(log_dir, data_dir)

        cb.on_train_start(trainer, pl_module=None)  # type: ignore[arg-type]

        copied_target = log_dir / "target_index.json"
        copied_meta = log_dir / "meta.json"
        assert copied_target.exists()
        assert copied_meta.exists()
        assert json.loads(copied_target.read_text()) == {"P12345": 0, "Q67890": 1}
        assert json.loads(copied_meta.read_text())["n_targets"] == 2

    def test_does_not_overwrite_existing(
        self, data_dir: Path, log_dir: Path
    ) -> None:
        (log_dir / "meta.json").write_text('{"already": "here"}')

        cb = CopyTrainingArtifacts()
        trainer = FakeTrainer(log_dir, data_dir)
        cb.on_train_start(trainer, pl_module=None)  # type: ignore[arg-type]

        assert json.loads((log_dir / "meta.json").read_text()) == {"already": "here"}
        assert (log_dir / "target_index.json").exists()

    def test_raises_if_source_missing(self, tmp_path: Path, log_dir: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        cb = CopyTrainingArtifacts()
        trainer = FakeTrainer(log_dir, empty_dir)

        with pytest.raises(FileNotFoundError):
            cb.on_train_start(trainer, pl_module=None)  # type: ignore[arg-type]

    def test_copies_oov_mapping_if_present(
        self, data_dir: Path, log_dir: Path
    ) -> None:
        (data_dir / "oov_target_mapping.json").write_text(
            json.dumps({"X11111": "P12345"})
        )

        cb = CopyTrainingArtifacts()
        trainer = FakeTrainer(log_dir, data_dir)
        cb.on_train_start(trainer, pl_module=None)  # type: ignore[arg-type]

        copied = log_dir / "oov_target_mapping.json"
        assert copied.exists()
        assert json.loads(copied.read_text()) == {"X11111": "P12345"}

    def test_skips_oov_mapping_if_absent(
        self, data_dir: Path, log_dir: Path
    ) -> None:
        cb = CopyTrainingArtifacts()
        trainer = FakeTrainer(log_dir, data_dir)
        cb.on_train_start(trainer, pl_module=None)  # type: ignore[arg-type]

        assert not (log_dir / "oov_target_mapping.json").exists()
