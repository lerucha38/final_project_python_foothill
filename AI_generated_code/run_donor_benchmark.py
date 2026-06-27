#!/usr/bin/env python3
"""Benchmark clonal inference tools on per-donor merged MiXCR repertoires."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from bcr_benchmark.compare import compare_all_methods, load_assignments, top_clones
from bcr_benchmark.config import (
    CONDA_RSCRIPT,
    DONOR_CLUSTERS_DIR,
    DONOR_COHORT_FRACTION,
    DONOR_COHORT_SIZE,
    DONOR_HILARY_METHOD,
    DONOR_METRICS_DIR,
    DONOR_PREPARED_DIR,
    HAMMING_THRESHOLDS,
    METHODS,
    PROJECT_ROOT,
    SYSTEM_RSCRIPT,
    parallel_env,
    resolve_nproc,
)
from bcr_benchmark.hamming_cluster import run_all_hamming_thresholds
from bcr_benchmark.memory import run_callable, run_subprocess
from bcr_benchmark.merge_donors import list_donors, prepare_merged_donor
from bcr_benchmark.plots import plot_cdr3_logo
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
    return result


def _method_stats(
    method: str,
    assignments: pd.DataFrame,
    runtime_sec: float,
    peak_rss_mb: float,
    n_sequences: int,
) -> dict:
    sizes_reads = assignments.groupby("clone_id")["duplicate_count"].sum()
    sizes_seqs = assignments.groupby("clone_id").size()
    return {
        "method": method,
        "runtime_sec": runtime_sec,
        "peak_rss_mb": peak_rss_mb,
        "n_sequences": n_sequences,
        "n_clone_groups": int(assignments["clone_id"].nunique()),
        "largest_clone_reads": int(sizes_reads.max()) if len(sizes_reads) else 0,
        "largest_clone_seqs": int(sizes_seqs.max()) if len(sizes_seqs) else 0,
    }


def _write_logo(method: str, assignments: pd.DataFrame, logo_dir: Path) -> None:
    if "junction_aa" not in assignments.columns:
        return
    largest = top_clones(assignments, n=1).iloc[0]["clone_id"]
    seqs = assignments.loc[assignments["clone_id"] == largest, "junction_aa"].astype(str).tolist()
    logo_dir.mkdir(parents=True, exist_ok=True)
    plot_cdr3_logo(
        seqs[:500],
        logo_dir / f"{method}_top_clone.png",
        title=f"{method} top clone ({len(seqs)} seqs)",
    )


def _load_prepared(donor_id: str, prepared_dir: Path, skip_existing: bool) -> tuple[pd.DataFrame, dict]:
    airr_path = prepared_dir / f"{donor_id}.airr.tsv"
    meta_path = prepared_dir / f"{donor_id}.meta.json"
    if skip_existing and airr_path.exists() and meta_path.exists():
        df = pd.read_csv(airr_path, sep="\t", low_memory=False)
        meta = json.loads(meta_path.read_text())
        return df, meta

    print(f"Merging timepoints for {donor_id}...")
    meta = prepare_merged_donor(donor_id, prepared_dir)
    df = pd.read_csv(airr_path, sep="\t", low_memory=False)
    return df, meta


def run_donor_benchmark(
    donor_id: str,
    methods: list[str] | None = None,
    nproc: int | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    methods = methods or METHODS
    workers = resolve_nproc(nproc)
    prepared_dir = DONOR_PREPARED_DIR
    cluster_dir = DONOR_CLUSTERS_DIR / donor_id
    metrics_dir = DONOR_METRICS_DIR / donor_id
    cluster_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df, meta = _load_prepared(donor_id, prepared_dir, skip_existing)
    airr_path = prepared_dir / f"{donor_id}.airr.tsv"
    hilary_path = prepared_dir / f"{donor_id}.hilary.tsv"
    n_sequences = len(df)

    if skip_existing and all((cluster_dir / f"{m}.tsv").exists() for m in methods):
        runtime_path = metrics_dir / "runtime.csv"
        print(f"Skipping {donor_id}: all method outputs already exist")
        if runtime_path.exists():
            return pd.read_csv(runtime_path)
        return pd.DataFrame()

    runtime_rows: list[dict] = []
    assignment_paths: dict[str, Path] = {}

    hamming_methods = [m for m in methods if m.startswith("hamming_")]
    pending_hamming = [m for m in hamming_methods if not (skip_existing and (cluster_dir / f"{m}.tsv").exists())]
    if pending_hamming:
        holder: dict = {}

        def _run_hamming() -> None:
            holder["results"] = run_all_hamming_thresholds(
                df, {m: HAMMING_THRESHOLDS[m] for m in pending_hamming}, nproc=workers
            )

        mem = run_callable(_run_hamming)
        for method_name, (clustered, hstats) in holder["results"].items():
            out_path = cluster_dir / f"{method_name}.tsv"
            clustered[["sequence_id", "clone_id"]].to_csv(out_path, sep="\t", index=False)
            assignment_paths[method_name] = out_path
            assigns = clustered[["sequence_id", "clone_id", "duplicate_count", "junction_aa"]]
            row = _method_stats(method_name, assigns, hstats["runtime_sec"], mem.peak_rss_mb, n_sequences)
            row["similarity_threshold"] = hstats["similarity_threshold"]
            runtime_rows.append(row)

    for method_name in hamming_methods:
        path = cluster_dir / f"{method_name}.tsv"
        if path.exists():
            assignment_paths[method_name] = path

    if "hilary" in methods:
        out_path = cluster_dir / "hilary.tsv"
        if skip_existing and out_path.exists():
            assignment_paths["hilary"] = out_path
        else:
            holder: dict = {}

            def _run_hilary() -> None:
                holder["result"] = run_hilary(
                    hilary_path,
                    cluster_dir / "hilary",
                    method=DONOR_HILARY_METHOD,
                    threads=workers,
                )

            mem = run_callable(_run_hilary)
            result_path, hstats = holder["result"]
            hilary_df = pd.read_csv(result_path, sep="\t", usecols=["sequence_id", "clone_id"])
            hilary_df["sequence_id"] = hilary_df["sequence_id"].astype(str).str.replace(r"-igh$", "", regex=True)
            hilary_df.to_csv(out_path, sep="\t", index=False)
            assignment_paths["hilary"] = out_path
            assigns = hilary_df.merge(
                df[["sequence_id", "duplicate_count", "junction_aa"]], on="sequence_id", how="left"
            )
            runtime_rows.append(
                _method_stats("hilary", assigns, hstats["runtime_sec"], mem.peak_rss_mb, n_sequences)
            )

    for scoper_method, mode in [
        ("scoper_hierarchical", "hierarchical"),
        ("scoper_spectral", "spectral"),
    ]:
        if scoper_method not in methods:
            continue
        out_path = cluster_dir / f"{scoper_method}.tsv"
        if skip_existing and out_path.exists():
            assignment_paths[scoper_method] = out_path
            continue
        stats = _run_r_script("scoper_runner.R", [str(airr_path), str(out_path), mode], scoper_method, workers)
        assignment_paths[scoper_method] = out_path
        assigns = load_assignments(out_path, scoper_method).merge(
            df[["sequence_id", "duplicate_count", "junction_aa"]], on="sequence_id"
        )
        runtime_rows.append(
            _method_stats(
                scoper_method, assigns, stats["runtime_sec"], stats["peak_rss_mb"], n_sequences
            )
        )

    if "fastbcr" in methods:
        out_path = cluster_dir / "fastbcr.tsv"
        if skip_existing and out_path.exists():
            assignment_paths["fastbcr"] = out_path
        elif not CONDA_RSCRIPT.exists():
            print(f"Skipping fastbcr for {donor_id}: conda R env missing")
        else:
            try:
                stats = _run_r_script("fastbcr_runner.R", [str(airr_path), str(out_path)], "fastbcr", workers)
                assignment_paths["fastbcr"] = out_path
                assigns = load_assignments(out_path, "fastbcr").merge(
                    df[["sequence_id", "duplicate_count", "junction_aa"]], on="sequence_id"
                )
                runtime_rows.append(
                    _method_stats(
                        "fastbcr", assigns, stats["runtime_sec"], stats["peak_rss_mb"], n_sequences
                    )
                )
            except RuntimeError as exc:
                print(f"fastbcr failed for {donor_id}: {exc}")

    assignments = {}
    for method, path in assignment_paths.items():
        if not path.exists():
            continue
        assignments[method] = load_assignments(path, method).merge(
            df[["sequence_id", "duplicate_count", "junction_aa"]], on="sequence_id"
        )

    newly_run = bool(runtime_rows)

    if not runtime_rows and assignments and not skip_existing:
        for method, assigns in assignments.items():
            runtime_rows.append(_method_stats(method, assigns, 0, 0, n_sequences))

    runtime_df = pd.DataFrame(runtime_rows)
    if not runtime_df.empty:
        runtime_df.insert(0, "donor_id", donor_id)
        runtime_df.to_csv(metrics_dir / "runtime.csv", index=False)

    metrics_paths = (
        metrics_dir / "pairwise_overlap.csv",
        metrics_dir / "clone_size_summary.csv",
    )
    if len(assignments) >= 2 and (
        newly_run or not all(p.exists() for p in metrics_paths)
    ):
        overlap_df, size_df = compare_all_methods(
            {m: d[["sequence_id", "clone_id", "duplicate_count"]] for m, d in assignments.items()}
        )
        overlap_df.insert(0, "donor_id", donor_id)
        size_df.insert(0, "donor_id", donor_id)
        overlap_df.to_csv(metrics_dir / "pairwise_overlap.csv", index=False)
        size_df.to_csv(metrics_dir / "clone_size_summary.csv", index=False)

    logo_dir = metrics_dir / "logos"
    for method, assigns in assignments.items():
        logo_path = logo_dir / f"{method}_top_clone.png"
        if skip_existing and logo_path.exists():
            continue
        _write_logo(method, assigns, logo_dir)

    summary = {"donor_id": donor_id, **meta, "nproc": workers, "methods_run": list(assignments.keys())}
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Finished {donor_id}: {len(assignments)} methods, {n_sequences:,} sequences")
    return runtime_df


def select_donor_cohort(
    size: int | None = DONOR_COHORT_SIZE,
    fraction: float = DONOR_COHORT_FRACTION,
) -> list[str]:
    donors = list_donors()
    if size is not None and size > 0:
        return donors[:size]
    n = max(1, int(len(donors) * fraction))
    return donors[:n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", help="Single donor ID")
    parser.add_argument("--cohort", action="store_true", help=f"Run first {DONOR_COHORT_SIZE} donors")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Parallel workers for all tools (default: 8; use 0 for all CPU cores)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DONOR_PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    DONOR_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if args.donor:
        donors = [args.donor]
    elif args.cohort:
        donors = select_donor_cohort()
        print(f"Running {len(donors)} donors: {donors[0]} ... {donors[-1]}")
    else:
        print("Specify --donor ID or --cohort", file=sys.stderr)
        return 1

    all_rows = []
    for donor_id in donors:
        try:
            row = run_donor_benchmark(
                donor_id,
                methods=args.methods,
                nproc=args.nproc,
                skip_existing=not args.force,
            )
            if row is not None and not row.empty:
                all_rows.append(row)
        except Exception as exc:
            print(f"ERROR {donor_id}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    if all_rows:
        master = pd.concat(all_rows, ignore_index=True)
        master_path = DONOR_METRICS_DIR / "all_donors_runtime.csv"
        if master_path.exists() and not args.force:
            existing = pd.read_csv(master_path)
            master = pd.concat([existing, master], ignore_index=True)
        master = master.drop_duplicates(subset=["donor_id", "method"], keep="last")
        master.to_csv(master_path, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
