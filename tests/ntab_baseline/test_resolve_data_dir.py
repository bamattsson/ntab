"""Tests for resolve_data_dir — finding training artifacts alongside a checkpoint."""

import json
from pathlib import Path

import pytest

from ntab_baseline.callbacks import resolve_data_dir


def _write_artifacts(d: Path) -> None:
    """Write the minimum required artifact files into directory d."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "target_index.json").write_text(json.dumps({"P12345": 0}))
    (d / "meta.json").write_text(json.dumps({"n_targets": 1}))


class TestResolveDataDir:
    def test_infers_version_dir_from_checkpoint(self, tmp_path: Path) -> None:
        version_dir = tmp_path / "lightning_logs" / "version_0"
        ckpt_dir = version_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "best.ckpt"
        ckpt.touch()
        _write_artifacts(version_dir)

        result = resolve_data_dir(ckpt)
        assert result == version_dir

    def test_raises_when_artifacts_missing(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt = ckpt_dir / "best.ckpt"
        ckpt.touch()

        with pytest.raises(FileNotFoundError, match="target_index.json"):
            resolve_data_dir(ckpt)
