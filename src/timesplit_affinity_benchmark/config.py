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
    one_assay_per_doc: bool


@dataclass
class PipelineConfig:
    tanimoto_threshold: (
        float | None
    )  # None disables the novelty filter (all candidates treated as novel)
    keep_discard_not_novel: bool
    year_val_start: int = 2022  # doc_year >= this is val; doc_year < this is train
    year_test_start: int = (
        2023  # doc_year >= this is test; year_val_start <= doc_year < this is val
    )
    n_jobs: int = 1
    activity_limit: int | None = None
    filter_val_and_test_sets: AssayFilterConfig | None = None


@dataclass
class Config:
    chembl_requester: ChEMBLConfig
    pipeline: PipelineConfig
    out_dir: str = (
        "out"  # root output directory; intermediate files go to {out_dir}/intermediate/
    )


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

    out_dir: str = raw.get("out_dir", "out")

    return Config(chembl_requester=chembl, pipeline=pipeline, out_dir=out_dir)
