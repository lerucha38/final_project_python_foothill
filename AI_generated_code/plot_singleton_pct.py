#!/usr/bin/env python3
"""Plot singleton-sequence fraction by method across BFI donors."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bcr_benchmark.config import DONOR_METRICS_DIR, RESULTS_DIR  # noqa: E402
from bcr_benchmark.plots import plot_singleton_pct_boxplot  # noqa: E402

FIGURES_DIR = RESULTS_DIR / "figures"
COMPOSITION_PATH = DONOR_METRICS_DIR / "all_donors_clone_composition.csv"


def main() -> None:
    if not COMPOSITION_PATH.exists():
        raise FileNotFoundError(
            f"Missing {COMPOSITION_PATH}. Run scripts/analyze_donor_clones.py first."
        )

    df = pd.read_csv(COMPOSITION_PATH)
    out_path = FIGURES_DIR / "singleton_pct_boxplot.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_singleton_pct_boxplot(df, out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
