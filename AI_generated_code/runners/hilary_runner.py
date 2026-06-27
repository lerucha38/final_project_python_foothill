"""Run HILARY clonal inference via CLI."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd


def run_hilary(
    input_path: Path,
    output_dir: Path,
    method: str = "full-method",
    threads: int = -1,
) -> tuple[Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "infer-lineages",
        method,
        str(input_path),
        "--result-folder",
        str(output_dir),
        "--threads",
        str(threads),
        "--override",
        "--silent",
    ]

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        raise RuntimeError(
            f"HILARY failed ({method}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    result_files = sorted(output_dir.glob("*.tsv"))
    if not result_files:
        raise FileNotFoundError(f"No HILARY output TSV found in {output_dir}")

    result_path = result_files[0]

    result_df = pd.read_csv(result_path, sep="\t", usecols=["sequence_id", "clone_id"])
    stats = {
        "method": f"hilary_{method.replace('-method', '')}",
        "runtime_sec": elapsed,
        "result_path": str(result_path),
        "n_sequences": len(result_df),
        "n_clone_groups": int(result_df["clone_id"].nunique()),
        "threads": threads,
        "stdout_tail": proc.stdout[-2000:],
    }
    return result_path, stats


def load_hilary_assignments(result_path: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(result_path, sep="\t", usecols=["sequence_id", "clone_id"])
    return df.set_index("sequence_id")["clone_id"].to_dict()
