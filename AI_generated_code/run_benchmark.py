#!/usr/bin/env python3
"""Benchmark BCR clonal family inference methods on bulk datasets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bcr_benchmark.compare import compare_all_methods, load_assignments
from bcr_benchmark.config import (
    CONDA_RSCRIPT,
    DATASETS,
    HAMMING_THRESHOLDS,
    METRICS_DIR,
    OUTPUT_DIR,
    PREPARED_DIR,
    PROJECT_ROOT,
    SYSTEM_RSCRIPT,
    parallel_env,
    resolve_nproc,
)
from bcr_benchmark.convert import load_dataset, prepare_dataset
from bcr_benchmark.hamming_cluster import run_all_hamming_thresholds
from bcr_benchmark.memory import run_subprocess
from bcr_benchmark.runners.hilary_runner import run_hilary


def _rscript_for(method: str) -> str:
    if method == "fastbcr" and CONDA_RSCRIPT.exists():
        return str(CONDA_RSCRIPT)
    return SYSTEM_RSCRIPT


def _r_env(for_fastbcr: bool = False, nproc: int = 1) -> dict[str, str]:
    base: dict[str, str] = {}
    if for_fastbcr and CONDA_RSCRIPT.exists():
        lib = str(CONDA_RSCRIPT.parent.parent / "lib" / "R" / "library")
        bin_dir = str(CONDA_RSCRIPT.parent)
        base.update(
            {
                "HOME": os.environ.get("HOME", "/tmp"),
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "R_LIBS_USER": lib,
                "R_LIBS": lib,
                "R_PROFILE_USER": "",
                "R_ENVIRON_USER": "",
            }
        )
    return parallel_env(nproc, base)


def _run_r_script(script_name: str, args: list[str], method: str, nproc: int) -> dict:
    script_path = PROJECT_ROOT / "bcr_benchmark" / "runners" / script_name
    use_conda = method == "fastbcr"
    cmd = [_rscript_for(method)]
    if use_conda:
        cmd.append("--vanilla")
    cmd.extend([str(script_path), *args])
    stats = run_subprocess(cmd, env=_r_env(for_fastbcr=use_conda, nproc=nproc))
    if stats.returncode != 0:
        raise RuntimeError(f"{script_name} failed with code {stats.returncode}")

    stats_path = Path(args[1]).with_suffix(".stats.json")
    result = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    result["runtime_sec"] = stats.runtime_sec
    result["peak_rss_mb"] = stats.peak_rss_mb
    result["nproc"] = nproc
    return result


def run_dataset_benchmark(
    dataset_key: str,
    methods: list[str] | None = None,
    nproc: int | None = None,
) -> None:
    cfg = DATASETS[dataset_key]
    workers = resolve_nproc(nproc)
    methods = methods or [
        "hamming_80",
        "hamming_90",
        "hamming_95",
        "hilary",
        "scoper_hierarchical",
        "scoper_spectral",
        "fastbcr",
    ]

    dataset_out = OUTPUT_DIR / dataset_key
    prepared_out = PREPARED_DIR / dataset_key
    metrics_out = METRICS_DIR / dataset_key
    dataset_out.mkdir(parents=True, exist_ok=True)
    metrics_out.mkdir(parents=True, exist_ok=True)

    prep = prepare_dataset(
        dataset_key,
        cfg["source_path"],
        cfg["source_format"],
        prepared_out,
    )
    airr_path = prep["airr"]
    hilary_path = prep["hilary"]
    df = load_dataset(cfg["source_path"], cfg["source_format"])

    runtime_rows = []
    assignment_paths: dict[str, Path] = {}

    if any(m.startswith("hamming_") for m in methods):
        hamming_methods = [m for m in methods if m.startswith("hamming_")]
        hamming_results = run_all_hamming_thresholds(
            df,
            {m: HAMMING_THRESHOLDS[m] for m in hamming_methods},
            nproc=workers,
        )
        for method_name, (clustered, stats) in hamming_results.items():
            out_path = dataset_out / f"{method_name}.tsv"
            clustered[["sequence_id", "clone_id"]].to_csv(out_path, sep="\t", index=False)
            assignment_paths[method_name] = out_path
            runtime_rows.append(stats)

    if "hilary" in methods:
        hilary_out = dataset_out / "hilary"
        result_path, stats = run_hilary(
            hilary_path,
            hilary_out,
            method=cfg["hilary_method"],
            threads=workers,
        )
        out_path = dataset_out / "hilary.tsv"
        hilary_df = pd.read_csv(result_path, sep="\t", usecols=["sequence_id", "clone_id"])
        hilary_df["sequence_id"] = hilary_df["sequence_id"].astype(str).str.replace(r"-igh$", "", regex=True)
        hilary_df.to_csv(out_path, sep="\t", index=False)
        assignment_paths["hilary"] = out_path
        stats["method"] = "hilary"
        stats["nproc"] = workers
        runtime_rows.append(stats)

    if "scoper_hierarchical" in methods:
        out_path = dataset_out / "scoper_hierarchical.tsv"
        stats = _run_r_script(
            "scoper_runner.R", [str(airr_path), str(out_path), "hierarchical"], "scoper_hierarchical", workers
        )
        assignment_paths["scoper_hierarchical"] = out_path
        stats["method"] = "scoper_hierarchical"
        runtime_rows.append(stats)

    if "scoper_spectral" in methods:
        out_path = dataset_out / "scoper_spectral.tsv"
        stats = _run_r_script(
            "scoper_runner.R", [str(airr_path), str(out_path), "spectral"], "scoper_spectral", workers
        )
        assignment_paths["scoper_spectral"] = out_path
        stats["method"] = "scoper_spectral"
        runtime_rows.append(stats)

    if "fastbcr" in methods:
        out_path = dataset_out / "fastbcr.tsv"
        stats = _run_r_script("fastbcr_runner.R", [str(airr_path), str(out_path)], "fastbcr", workers)
        assignment_paths["fastbcr"] = out_path
        stats["method"] = "fastbcr"
        runtime_rows.append(stats)

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(metrics_out / "runtime.csv", index=False)

    assignments = {
        method: load_assignments(path, method).merge(
            df[["sequence_id", "duplicate_count"]], on="sequence_id", how="left"
        )
        for method, path in assignment_paths.items()
    }

    overlap_df, size_df = compare_all_methods(assignments)
    overlap_df.to_csv(metrics_out / "pairwise_overlap.csv", index=False)
    size_df.to_csv(metrics_out / "clone_size_summary.csv", index=False)

    summary = {
        "dataset": dataset_key,
        "label": cfg["label"],
        "n_sequences": len(df),
        "nproc": workers,
        "methods_run": list(assignment_paths.keys()),
        "prepared_files": {k: str(v) for k, v in prep.items() if k != "n_sequences"},
    }
    (metrics_out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {dataset_key} complete (nproc={workers}) ===")
    display_cols = [c for c in ["method", "runtime_sec", "n_clone_groups", "n_sequences"] if c in runtime_df.columns]
    print(runtime_df[display_cols].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Parallel workers for all tools (default: all CPU cores; use 0 for same)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from bcr_benchmark.config import RESULTS_DIR

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for dataset_key in datasets:
        run_dataset_benchmark(dataset_key, methods=args.methods, nproc=args.nproc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
