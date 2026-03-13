from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ChEMBLConfig:
    host: str
    user: str
    password: str
    dbname: str


@dataclass
class PipelineConfig:
    tanimoto_threshold: float
    keep_not_novel: bool
    n_jobs: int = 1
    activity_limit: int | None = None


@dataclass
class Config:
    chembl_requester: ChEMBLConfig
    pipeline: PipelineConfig


def load_config(path: str | Path) -> Config:
    """Load and validate pipeline configuration from a YAML file.

    Args:
        path: Path to the config YAML file.

    Returns:
        Parsed Config object.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    chembl = ChEMBLConfig(**raw["chembl_requester"])
    pipeline = PipelineConfig(**raw.get("pipeline", {}))

    return Config(chembl_requester=chembl, pipeline=pipeline)
