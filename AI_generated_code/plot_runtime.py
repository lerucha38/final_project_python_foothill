#!/usr/bin/env python3
"""Collect benchmark runtimes and plot boxplots by method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bcr_benchmark.config import (  # noqa: E402
    DATASETS,
    DONOR_COHORT_SIZE,
    DONOR_PREPARED_DIR,
    HAMMING_THRESHOLDS,
    METHODS,
    PREPARED_DIR,
    RESULTS_DIR,
)
from bcr_benchmark.hamming_cluster import cluster_hamming  # noqa: E402
from bcr_benchmark.plots import plot_runtime_boxplot  # noqa: E402
from bcr_benchmark.runners.hilary_runner import run_hilary  # noqa: E402

CLUSTERS_DIR = RESULTS_DIR / "clusters"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNTIME_CACHE = METRICS_DIR / "all_runtimes.csv"
MERGED_RUNTIME_CACHE = METRICS_DIR / "merged_runtimes.csv"
PLOT_TITLE = "Runtime difference between different clonal inference approaches"

# Restored timings lost when metrics were regenerated without re-running tools.
RUNTIME_OVERRIDES: list[dict] = [
    {
        "dataset_id": "BFI-0000372",
        "method": "hilary",
        "runtime_sec": 113.0,
        "n_sequences": 785942,
        "source": "original_run:restored",
    },
]


def is_merged_dataset(dataset_id: str) -> bool:
    """Include donor merged timepoints only (BFI-*)."""
    return dataset_id.startswith("BFI-")


def _donor_ids() -> list[str]:
    cohort_file = RESULTS_DIR / "donors" / "cohort_donors.txt"
    if cohort_file.exists():
        ids = [line.strip() for line in cohort_file.read_text().splitlines() if line.strip()]
        return ids[:DONOR_COHORT_SIZE]
    return sorted(p.stem for p in DONOR_PREPARED_DIR.glob("BFI-*.airr.tsv"))[:DONOR_COHORT_SIZE]


def _load_prepared(dataset_id: str) -> pd.DataFrame:
    if dataset_id.startswith("BFI-"):
        path = DONOR_PREPARED_DIR / f"{dataset_id}.airr.tsv"
    else:
        path = PREPARED_DIR / dataset_id / f"{dataset_id}.airr.tsv"
    return pd.read_csv(path, sep="\t", low_memory=False)


def collect_from_stats() -> list[dict]:
    rows: list[dict] = []
    for path in CLUSTERS_DIR.rglob("*.stats.json"):
        rel = path.relative_to(CLUSTERS_DIR)
        parts = rel.parts
        if parts[0] == "donors":
            dataset_id = parts[1]
        else:
            dataset_id = parts[0]
        payload = json.loads(path.read_text())
        runtime = payload.get("runtime_sec")
        if runtime is None or runtime <= 0:
            continue
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": payload.get("method", path.stem.replace(".stats", "")),
                "runtime_sec": float(runtime),
                "n_sequences": int(payload.get("n_sequences", 0)),
                "source": str(path),
            }
        )
    return rows


def collect_from_runtime_csv() -> list[dict]:
    rows: list[dict] = []
    for path in METRICS_DIR.rglob("runtime.csv"):
        df = pd.read_csv(path)
        if "runtime_sec" not in df.columns:
            continue
        df = df[df["runtime_sec"] > 0]
        if df.empty:
            continue
        if "donor_id" in df.columns:
            for _, row in df.iterrows():
                rows.append(
                    {
                        "dataset_id": row["donor_id"],
                        "method": row["method"],
                        "runtime_sec": float(row["runtime_sec"]),
                        "n_sequences": int(row.get("n_sequences", 0)),
                        "source": str(path),
                    }
                )
        else:
            dataset_id = path.parent.name
            for _, row in df.iterrows():
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": row["method"],
                        "runtime_sec": float(row["runtime_sec"]),
                        "n_sequences": int(row.get("n_sequences", 0)),
                        "source": str(path),
                    }
                )
    return rows


def dedupe_records(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["dataset_id", "method", "runtime_sec", "n_sequences", "source"])
    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset_id", "method", "runtime_sec"])
    return df.drop_duplicates(subset=["dataset_id", "method"], keep="last").reset_index(drop=True)


def collect_runtime_records() -> pd.DataFrame:
    rows = collect_from_stats() + collect_from_runtime_csv()
    return dedupe_records(rows)


def backfill_hamming(nproc: int = 8) -> list[dict]:
    rows: list[dict] = []
    dataset_ids = list(DATASETS) + _donor_ids()
    for dataset_id in dataset_ids:
        df = _load_prepared(dataset_id)
        n_sequences = len(df)
        for method, similarity in HAMMING_THRESHOLDS.items():
            _, stats = cluster_hamming(df, similarity=similarity, nproc=nproc)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "runtime_sec": stats["runtime_sec"],
                    "n_sequences": n_sequences,
                    "source": "backfill:hamming",
                }
            )
            print(f"{dataset_id} {method}: {stats['runtime_sec']:.1f}s ({n_sequences} seqs)")
    return rows


def backfill_hilary(nproc: int = 8) -> list[dict]:
    rows: list[dict] = []
    jobs: list[tuple[str, Path, Path, str]] = []
    for dataset_id, cfg in DATASETS.items():
        hilary_path = PREPARED_DIR / dataset_id / f"{dataset_id}.hilary.tsv"
        out_dir = CLUSTERS_DIR / dataset_id / "hilary_backfill"
        jobs.append((dataset_id, hilary_path, out_dir, cfg["hilary_method"]))
    for donor_id in _donor_ids():
        hilary_path = DONOR_PREPARED_DIR / f"{donor_id}.hilary.tsv"
        out_dir = CLUSTERS_DIR / "donors" / donor_id / "hilary_backfill"
        jobs.append((donor_id, hilary_path, out_dir, "cdr3-method"))

    for dataset_id, hilary_path, out_dir, method in jobs:
        if not hilary_path.exists():
            print(f"Skipping HILARY backfill for {dataset_id}: missing {hilary_path}")
            continue
        n_sequences = sum(1 for _ in open(hilary_path)) - 1
        _, stats = run_hilary(hilary_path, out_dir, method=method, threads=nproc)
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": "hilary",
                "runtime_sec": stats["runtime_sec"],
                "n_sequences": n_sequences,
                "source": "backfill:hilary",
            }
        )
        print(f"{dataset_id} hilary: {stats['runtime_sec']:.1f}s ({n_sequences} seqs)")
    return rows


def apply_runtime_overrides(records: pd.DataFrame) -> pd.DataFrame:
    if not RUNTIME_OVERRIDES:
        return records
    merged = pd.concat([records, pd.DataFrame(RUNTIME_OVERRIDES)], ignore_index=True)
    return dedupe_records(merged.to_dict("records"))


def merge_records(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return existing
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    return dedupe_records(combined.to_dict("records"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark runtime boxplots.")
    parser.add_argument(
        "--backfill",
        choices=("none", "hamming", "hilary", "all"),
        default="hamming",
        help="Re-time methods missing from saved stats (default: hamming only).",
    )
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "runtime_boxplot.png",
    )
    args = parser.parse_args()

    records = collect_runtime_records()
    missing_hamming = {
        (dataset_id, method)
        for dataset_id in list(DATASETS) + _donor_ids()
        for method in HAMMING_THRESHOLDS
    } - set(zip(records["dataset_id"], records["method"]))
    missing_hilary = {
        (dataset_id, "hilary")
        for dataset_id in list(DATASETS) + _donor_ids()
    } - set(zip(records["dataset_id"], records["method"]))

    if args.backfill in ("hamming", "all") and missing_hamming:
        records = merge_records(records, backfill_hamming(nproc=args.nproc))
    if args.backfill in ("hilary", "all") and missing_hilary:
        records = merge_records(records, backfill_hilary(nproc=args.nproc))

    records = records[records["method"].isin(METHODS)].copy()
    records = records[records["runtime_sec"] > 0].copy()
    RUNTIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(RUNTIME_CACHE, index=False)

    plot_records = records[records["dataset_id"].map(is_merged_dataset)].copy()
    plot_records = apply_runtime_overrides(plot_records)
    plot_records.to_csv(MERGED_RUNTIME_CACHE, index=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_runtime_boxplot(plot_records, args.output, title=PLOT_TITLE)
    print(f"Saved {len(records)} runtime records to {RUNTIME_CACHE}")
    print(f"Saved {len(plot_records)} merged-dataset records to {MERGED_RUNTIME_CACHE}")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
