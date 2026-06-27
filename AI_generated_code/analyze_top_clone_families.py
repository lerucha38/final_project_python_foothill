#!/usr/bin/env python3
"""Match top-N clones across methods and compare shared family sizes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bcr_benchmark.compare import find_top_clone_shared_families, load_assignments  # noqa: E402
from bcr_benchmark.config import (  # noqa: E402
    DONOR_CLUSTERS_DIR,
    DONOR_COHORT_SIZE,
    DONOR_METRICS_DIR,
    DONOR_PREPARED_DIR,
    METHODS,
    RESULTS_DIR,
)
from bcr_benchmark.plots import plot_top_clone_family_sizes  # noqa: E402

DEFAULT_TOP_N = 10
DEFAULT_MIN_OVERLAP = 0.5


def _donor_ids() -> list[str]:
    cohort_file = RESULTS_DIR / "donors" / "cohort_donors.txt"
    if cohort_file.exists():
        ids = [line.strip() for line in cohort_file.read_text().splitlines() if line.strip()]
        return ids[:DONOR_COHORT_SIZE]
    return sorted(p.stem for p in DONOR_PREPARED_DIR.glob("BFI-*.airr.tsv"))[:DONOR_COHORT_SIZE]


def _load_assignments(donor_id: str, methods: list[str]) -> dict[str, pd.DataFrame]:
    assignments: dict[str, pd.DataFrame] = {}
    for method in methods:
        cluster_path = DONOR_CLUSTERS_DIR / donor_id / f"{method}.tsv"
        if not cluster_path.exists():
            raise FileNotFoundError(f"Missing cluster file: {cluster_path}")
        assignments[method] = load_assignments(cluster_path, method)
    return assignments


def _load_junction_aa(donor_id: str) -> dict[str, str]:
    path = DONOR_PREPARED_DIR / f"{donor_id}.airr.tsv"
    seqs = pd.read_csv(path, sep="\t", usecols=["sequence_id", "junction_aa"], low_memory=False)
    return dict(zip(seqs["sequence_id"], seqs["junction_aa"].astype(str)))


def analyze_donor(
    donor_id: str,
    methods: list[str],
    top_n: int,
    min_overlap: float,
) -> pd.DataFrame:
    assignments = _load_assignments(donor_id, methods)
    junction_aa = _load_junction_aa(donor_id)
    families = find_top_clone_shared_families(
        assignments,
        methods=methods,
        top_n=top_n,
        min_overlap=min_overlap,
        junction_aa=junction_aa,
    )
    if families.empty:
        return families
    families.insert(0, "donor_id", donor_id)
    return families


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare top-N clone assignments across methods and find shared families."
    )
    parser.add_argument("--donor", action="append", help="Donor ID (repeatable). Default: cohort list.")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    donors = args.donor or _donor_ids()
    all_rows: list[pd.DataFrame] = []

    for donor_id in donors:
        print(f"Analyzing {donor_id}...")
        df = analyze_donor(donor_id, args.methods, args.top_n, args.min_overlap)
        if df.empty:
            print(f"  no shared families found")
            continue

        out_dir = DONOR_METRICS_DIR / donor_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "top_clone_shared_families.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({df['family_id'].nunique()} families)")
        all_rows.append(df)

    if not all_rows:
        print("No shared families found for any donor.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = DONOR_METRICS_DIR / "all_donors_top_clone_shared_families.csv"
    combined.to_csv(combined_path, index=False)
    print(
        f"Wrote {combined_path} "
        f"({combined['donor_id'].nunique()} donors, "
        f"{combined.groupby(['donor_id', 'family_id']).ngroups} shared families, "
        f"{len(combined)} rows)"
    )

    if not args.skip_plot:
        figure_path = RESULTS_DIR / "figures" / "top_clone_shared_families.png"
        plot_top_clone_family_sizes(combined, figure_path)
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
