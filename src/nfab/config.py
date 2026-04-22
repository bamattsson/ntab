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
class SimilarityBin:
    """Defines one similarity bin for test/val split assignment.

    Use (low, hi) for a half-open range [low, hi), or equal for an exact match.
    """

    low: float | None = None
    hi: float | None = None
    equal: float | None = None


@dataclass
class AssayFilterConfig:
    apply_to: list[str]
    only_equal_relation: bool
    min_std: float
    min_cpd_per_assay: int
    one_assay_per_doc: bool


@dataclass
class PipelineConfig:
    test_set_similarity_bins: list[SimilarityBin]
    split_val_like_test: bool = True  # if True, val gets same sim bins as test; if False, val is a single "val" split
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

    bins_raw = pipeline_raw.pop("test_set_similarity_bins", [])
    similarity_bins = [SimilarityBin(**b) for b in bins_raw]

    pipeline = PipelineConfig(
        **pipeline_raw,
        test_set_similarity_bins=similarity_bins,
        filter_val_and_test_sets=filter_config,
    )

    out_dir: str = raw.get("out_dir", "out")

    return Config(chembl_requester=chembl, pipeline=pipeline, out_dir=out_dir)
