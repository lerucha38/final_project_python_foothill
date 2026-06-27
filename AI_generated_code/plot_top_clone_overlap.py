#!/usr/bin/env python3
"""Plot top-N clone overlap counts by how many methods each group spans."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bcr_benchmark.compare import count_top_clone_method_coverage, load_assignments  # noqa: E402
from bcr_benchmark.config import (  # noqa: E402
    DONOR_CLUSTERS_DIR,
    DONOR_COHORT_SIZE,
    DONOR_METRICS_DIR,
    METHODS,
    RESULTS_DIR,
)
from bcr_benchmark.plots import plot_top_clone_method_coverage  # noqa: E402

DEFAULT_TOP_N = 10
DEFAULT_MIN_OVERLAP = 0.5


def _donor_ids() -> list[str]:
    cohort_file = RESULTS_DIR / "donors" / "cohort_donors.txt"
    if cohort_file.exists():
        ids = [line.strip() for line in cohort_file.read_text().splitlines() if line.strip()]
        return ids[:DONOR_COHORT_SIZE]
    return sorted(p.stem for p in DONOR_METRICS_DIR.glob("BFI-*/runtime.csv"))[:DONOR_COHORT_SIZE]


def _load_assignments(donor_id: str, methods: list[str]) -> dict[str, pd.DataFrame]:
    assignments: dict[str, pd.DataFrame] = {}
    for method in methods:
        cluster_path = DONOR_CLUSTERS_DIR / donor_id / f"{method}.tsv"
        if not cluster_path.exists():
            raise FileNotFoundError(f"Missing cluster file: {cluster_path}")
        assignments[method] = load_assignments(cluster_path, method)
    return assignments


def analyze_donor(
    donor_id: str,
    methods: list[str],
    top_n: int,
    min_overlap: float,
) -> pd.DataFrame:
    assignments = _load_assignments(donor_id, methods)
    counts = count_top_clone_method_coverage(
        assignments,
        methods=methods,
        top_n=top_n,
        min_overlap=min_overlap,
    )
    if counts.empty:
        return counts
    counts.insert(0, "donor_id", donor_id)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count top-N clone groups by method coverage and plot across donors."
    )
    parser.add_argument("--donor", action="append", help="Donor ID (repeatable). Default: cohort list.")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP)
    args = parser.parse_args()

    donors = args.donor or _donor_ids()
    all_rows: list[pd.DataFrame] = []

    for donor_id in donors:
        print(f"Analyzing {donor_id}...")
        df = analyze_donor(donor_id, args.methods, args.top_n, args.min_overlap)
        if df.empty:
            print("  no top-clone records")
            continue

        out_dir = DONOR_METRICS_DIR / donor_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "top_clone_method_coverage.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")
        all_rows.append(df)

    if not all_rows:
        raise SystemExit("No coverage records produced.")

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["donor_id", "n_methods_found"], keep="last")
    combined_path = DONOR_METRICS_DIR / "all_donors_top_clone_method_coverage.csv"
    combined.to_csv(combined_path, index=False)

    figure_path = RESULTS_DIR / "figures" / "top_clone_method_coverage.png"
    plot_top_clone_method_coverage(combined, figure_path)

    print(f"Wrote {combined_path} ({len(combined)} rows)")
    print(f"Wrote {figure_path}")
    print("\nSample (first donor):")
    print(combined[combined["donor_id"] == combined["donor_id"].iloc[0]].to_string(index=False))


if __name__ == "__main__":
    main()
