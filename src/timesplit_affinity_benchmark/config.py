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
class AssayFilterConfig:
    apply_to: list[str]
    only_equal_relation: bool
    min_std: float
    min_cpd_per_assay: int
    one_assay_per_doi: bool


@dataclass
class PipelineConfig:
    tanimoto_threshold: float
    keep_not_novel_in_test: bool
    n_jobs: int = 1
    activity_limit: int | None = None
    filter_val_and_test_sets: AssayFilterConfig | None = None


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

    pipeline_raw = dict(raw.get("pipeline", {}))
    filter_raw = pipeline_raw.pop("filter_val_and_test_sets", None)
    filter_config = AssayFilterConfig(**filter_raw) if filter_raw else None

    pipeline = PipelineConfig(**pipeline_raw, filter_val_and_test_sets=filter_config)

    return Config(chembl_requester=chembl, pipeline=pipeline)
