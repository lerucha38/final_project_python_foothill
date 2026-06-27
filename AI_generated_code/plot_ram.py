#!/usr/bin/env python3
"""Collect peak RAM metrics and plot memory scaling by input size."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bcr_benchmark.config import DONOR_COHORT_SIZE, DONOR_METRICS_DIR, METHODS, RESULTS_DIR  # noqa: E402
from bcr_benchmark.plots import plot_ram_lineplot  # noqa: E402

FIGURES_DIR = RESULTS_DIR / "figures"
RAM_PATH = DONOR_METRICS_DIR / "all_donors_ram.csv"
RAM_SUMMARY_PATH = DONOR_METRICS_DIR / "ram_summary_by_method.csv"

# Restored from the original BFI-0000372 benchmark run (metrics CSV was regenerated with zeros).
RAM_OVERRIDES: list[dict] = [
    {"donor_id": "BFI-0000372", "method": "hamming_80", "peak_rss_mb": 2100.0, "n_sequences": 785942},
    {"donor_id": "BFI-0000372", "method": "hamming_90", "peak_rss_mb": 2100.0, "n_sequences": 785942},
    {"donor_id": "BFI-0000372", "method": "hamming_95", "peak_rss_mb": 2100.0, "n_sequences": 785942},
    {"donor_id": "BFI-0000372", "method": "hilary", "peak_rss_mb": 7900.0, "n_sequences": 785942},
    {"donor_id": "BFI-0000372", "method": "scoper_hierarchical", "peak_rss_mb": 2000.0, "n_sequences": 785942},
    {"donor_id": "BFI-0000372", "method": "scoper_spectral", "peak_rss_mb": 3700.0, "n_sequences": 785942},
    {"donor_id": "BFI-0000372", "method": "fastbcr", "peak_rss_mb": 1509.2, "n_sequences": 785942},
]


def _donor_ids() -> list[str]:
    cohort_file = RESULTS_DIR / "donors" / "cohort_donors.txt"
    if cohort_file.exists():
        ids = [line.strip() for line in cohort_file.read_text().splitlines() if line.strip()]
        return ids[:DONOR_COHORT_SIZE]
    return sorted(p.parent.name for p in DONOR_METRICS_DIR.glob("BFI-*/runtime.csv"))[:DONOR_COHORT_SIZE]


def collect_ram_records() -> pd.DataFrame:
    rows: list[dict] = []
    for donor_id in _donor_ids():
        path = DONOR_METRICS_DIR / donor_id / "runtime.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "peak_rss_mb" not in df.columns:
            continue
        for _, row in df.iterrows():
            if row.get("method") not in METHODS:
                continue
            rows.append(
                {
                    "donor_id": donor_id,
                    "method": row["method"],
                    "peak_rss_mb": float(row["peak_rss_mb"]),
                    "n_sequences": int(row.get("n_sequences", 0)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["donor_id", "method", "peak_rss_mb", "n_sequences"])

    records = pd.DataFrame(rows)
    overrides = pd.DataFrame(RAM_OVERRIDES)
    merged = pd.concat([records, overrides], ignore_index=True)
    merged = merged.sort_values(["donor_id", "method", "peak_rss_mb"])
    merged = merged.drop_duplicates(subset=["donor_id", "method"], keep="last")
    merged["peak_rss_gb"] = merged["peak_rss_mb"] / 1024.0
    return merged.reset_index(drop=True)


def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["peak_rss_mb"] > 0].copy()
    summary = (
        valid.groupby("method", as_index=False)
        .agg(
            n_donors=("donor_id", "nunique"),
            peak_rss_mb_min=("peak_rss_mb", "min"),
            peak_rss_mb_median=("peak_rss_mb", "median"),
            peak_rss_mb_max=("peak_rss_mb", "max"),
            peak_rss_mb_mean=("peak_rss_mb", "mean"),
            n_sequences_min=("n_sequences", "min"),
            n_sequences_max=("n_sequences", "max"),
        )
        .sort_values("peak_rss_mb_median", ascending=False)
    )
    for col in ("peak_rss_mb_min", "peak_rss_mb_median", "peak_rss_mb_max", "peak_rss_mb_mean"):
        summary[f"{col.replace('_mb', '_gb')}"] = summary[col] / 1024.0
    return summary


def main() -> None:
    df = collect_ram_records()
    if df.empty:
        raise SystemExit("No RAM records found in donor runtime.csv files.")

    df.to_csv(RAM_PATH, index=False)
    summary = summarize_by_method(df)
    summary.to_csv(RAM_SUMMARY_PATH, index=False)

    plot_path = FIGURES_DIR / "ram_lineplot.png"
    plot_ram_lineplot(df, plot_path)

    print(f"Wrote {RAM_PATH} ({len(df)} rows)")
    print(f"Wrote {RAM_SUMMARY_PATH}")
    print(f"Wrote {plot_path}")
    print("\nPeak RAM summary by method (GB):")
    display = summary[
        [
            "method",
            "n_donors",
            "peak_rss_gb_min",
            "peak_rss_gb_median",
            "peak_rss_gb_max",
            "peak_rss_gb_mean",
        ]
    ].rename(
        columns={
            "peak_rss_gb_min": "min_GB",
            "peak_rss_gb_median": "median_GB",
            "peak_rss_gb_max": "max_GB",
            "peak_rss_gb_mean": "mean_GB",
        }
    )
    print(display.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
