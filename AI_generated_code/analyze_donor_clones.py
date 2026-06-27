#!/usr/bin/env python3
"""Per-donor clone composition stats and top-10 CDR3 logos for each method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bcr_benchmark.compare import clone_composition_stats, load_assignments, top_clones  # noqa: E402
from bcr_benchmark.config import (  # noqa: E402
    DONOR_CLUSTERS_DIR,
    DONOR_COHORT_SIZE,
    DONOR_METRICS_DIR,
    DONOR_PREPARED_DIR,
    METHODS,
    RESULTS_DIR,
)
from bcr_benchmark.plots import plot_cdr3_logo  # noqa: E402

TOP_N = 10
MAX_LOGO_SEQS = 500


def _donor_ids() -> list[str]:
    cohort_file = RESULTS_DIR / "donors" / "cohort_donors.txt"
    if cohort_file.exists():
        ids = [line.strip() for line in cohort_file.read_text().splitlines() if line.strip()]
        return ids[:DONOR_COHORT_SIZE]
    return sorted(p.stem for p in DONOR_PREPARED_DIR.glob("BFI-*.airr.tsv"))[:DONOR_COHORT_SIZE]


def _load_donor_sequences(donor_id: str) -> pd.DataFrame:
    path = DONOR_PREPARED_DIR / f"{donor_id}.airr.tsv"
    return pd.read_csv(path, sep="\t", usecols=["sequence_id", "junction_aa"], low_memory=False)


def _load_method_assignments(donor_id: str, method: str) -> pd.DataFrame | None:
    cluster_path = DONOR_CLUSTERS_DIR / donor_id / f"{method}.tsv"
    if not cluster_path.exists():
        return None
    assigns = load_assignments(cluster_path, method)
    seqs = _load_donor_sequences(donor_id)
    return assigns.merge(seqs, on="sequence_id", how="left")


def _write_top_logos(method: str, assignments: pd.DataFrame, logo_dir: Path) -> None:
    if "junction_aa" not in assignments.columns:
        return
    logo_dir.mkdir(parents=True, exist_ok=True)
    ranked = top_clones(assignments, n=TOP_N, weight_col=None)
    for rank, row in enumerate(ranked.itertuples(), start=1):
        clone_id = row.clone_id
        size = int(row.clone_size)
        seqs = (
            assignments.loc[assignments["clone_id"] == clone_id, "junction_aa"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if not seqs:
            continue
        plot_cdr3_logo(
            seqs[:MAX_LOGO_SEQS],
            logo_dir / f"top_{rank:02d}.png",
            title=f"{method} rank {rank} (n={size} seqs)",
        )


def analyze_donor(donor_id: str, methods: list[str], skip_logos: bool = False) -> pd.DataFrame:
    metrics_dir = DONOR_METRICS_DIR / donor_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for method in methods:
        assignments = _load_method_assignments(donor_id, method)
        if assignments is None:
            print(f"  skip {method}: no cluster file")
            continue

        stats = clone_composition_stats(assignments)
        stats["donor_id"] = donor_id
        stats["method"] = method
        rows.append(stats)

        if not skip_logos:
            _write_top_logos(method, assignments, metrics_dir / "logos" / method)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    col_order = [
        "donor_id",
        "method",
        "n_total_sequences",
        "n_total_clones",
        "n_clones_size_1",
        "n_clones_size_2",
        "n_sequences_in_size_1_clones",
        "n_sequences_in_size_2_clones",
        "pct_sequences_in_size_1_clones",
        "pct_sequences_in_size_2_clones",
    ]
    df = df[col_order]
    df.to_csv(metrics_dir / "clone_composition.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone composition and top-10 logos per donor.")
    parser.add_argument("--donor", action="append", help="Donor ID (repeatable). Default: cohort list.")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--skip-logos", action="store_true")
    args = parser.parse_args()

    donors = args.donor or _donor_ids()
    all_rows: list[pd.DataFrame] = []

    for donor_id in donors:
        print(f"Analyzing {donor_id}...")
        df = analyze_donor(donor_id, args.methods, skip_logos=args.skip_logos)
        if not df.empty:
            all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_path = DONOR_METRICS_DIR / "all_donors_clone_composition.csv"
        combined.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
