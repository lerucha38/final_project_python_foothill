import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_nproc(nproc: int | None = None) -> int:
    """Resolve parallel worker count.

    Default is DEFAULT_NPROC (8). Pass <=0 to use all available CPU cores.
    """
    if nproc is None:
        return DEFAULT_NPROC
    if nproc <= 0:
        return os.cpu_count() or 1
    return nproc


def parallel_env(nproc: int, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment variables for BLAS/OpenMP-backed libraries."""
    env = dict(os.environ)
    if base:
        env.update(base)
    n = str(resolve_nproc(nproc))
    env["BENCHMARK_NPROC"] = n
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = n
    return env


DEFAULT_NPROC = 8
RESULTS_DIR = PROJECT_ROOT / "results"
PREPARED_DIR = RESULTS_DIR / "prepared"
OUTPUT_DIR = RESULTS_DIR / "clusters"
METRICS_DIR = RESULTS_DIR / "metrics"

DONOR_RESULTS_DIR = RESULTS_DIR / "donors"
DONOR_PREPARED_DIR = PREPARED_DIR / "donors"
DONOR_CLUSTERS_DIR = OUTPUT_DIR / "donors"
DONOR_METRICS_DIR = METRICS_DIR / "donors"

MIXCR_DIR = Path("/home/vskatova/ellison_bcr_R_data_files/result_mixcr_cDNA")

# Conda R env used for fastBCR (system R used for SCOPer)
CONDA_RSCRIPT = PROJECT_ROOT / "envs" / "bcr-r" / "bin" / "Rscript"
SYSTEM_RSCRIPT = "Rscript"

DATASETS = {
    "bulk_single": {
        "label": "Bulk single library (BFI-0010813 2011 V1)",
        "source_path": Path(
            "/home/vskatova/ellison_bcr_R_data_files/result_mixcr_cDNA/"
            "BFI-0010813_2011_V1.clones_IGH.tsv"
        ),
        "source_format": "mixcr",
        "hilary_method": "cdr3-method",
    },
    "bulk_merged": {
        "label": "Merged bulk (28 samples, BFI-0010813)",
        "source_path": Path("/home/amikelov/ellison/hilary/BFI-0010813.merged.tsv"),
        "source_format": "hilary",
        "hilary_method": "full-method",
    },
}

HAMMING_THRESHOLDS = {
    "hamming_80": 0.80,
    "hamming_90": 0.90,
    "hamming_95": 0.95,
}

METHODS = [
    "hamming_80",
    "hamming_90",
    "hamming_95",
    "hilary",
    "scoper_hierarchical",
    "scoper_spectral",
    "fastbcr",
]

SCOPER_HIERARCHICAL_THRESHOLD = 0.15
DONOR_HILARY_METHOD = "cdr3-method"
DONOR_COHORT_SIZE = 10
DONOR_COHORT_FRACTION = 0.5  # legacy; cohort uses DONOR_COHORT_SIZE when set
