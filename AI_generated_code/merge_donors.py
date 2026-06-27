"""Merge MiXCR clone tables across timepoints per donor."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from bcr_benchmark.convert import load_mixcr, to_hilary_table

MIXCR_DIR = Path("/home/vskatova/ellison_bcr_R_data_files/result_mixcr_cDNA")
DONOR_PATTERN = "BFI-*"


def list_donors(mixcr_dir: Path = MIXCR_DIR) -> list[str]:
    donors = sorted({f.name.split("_")[0] for f in mixcr_dir.glob("*.clones_IGH.tsv")})
    return donors


def donor_files(donor_id: str, mixcr_dir: Path = MIXCR_DIR) -> list[Path]:
    return sorted(mixcr_dir.glob(f"{donor_id}_*.clones_IGH.tsv"))


def merge_donor(donor_id: str, mixcr_dir: Path = MIXCR_DIR) -> pd.DataFrame:
    frames = []
    for path in donor_files(donor_id, mixcr_dir):
        sample = path.name.replace(".clones_IGH.tsv", "")
        chunk = load_mixcr(path)
        chunk["sample_id"] = sample
        chunk["sequence_id"] = sample + "::" + chunk["sequence_id"]
        frames.append(chunk)

    if not frames:
        raise FileNotFoundError(f"No MiXCR files found for donor {donor_id}")

    merged = pd.concat(frames, ignore_index=True)
    return merged


def prepare_merged_donor(
    donor_id: str,
    output_dir: Path,
    mixcr_dir: Path = MIXCR_DIR,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = merge_donor(donor_id, mixcr_dir)

    airr_path = output_dir / f"{donor_id}.airr.tsv"
    hilary_path = output_dir / f"{donor_id}.hilary.tsv"
    meta_path = output_dir / f"{donor_id}.meta.json"

    df.to_csv(airr_path, sep="\t", index=False)
    to_hilary_table(df, include_alignments=False).to_csv(hilary_path, sep="\t", index=False, na_rep="")

    import json

    meta = {
        "donor_id": donor_id,
        "n_sequences": len(df),
        "n_samples": int(df["sample_id"].nunique()),
        "n_unique_cdr3": int(df["junction"].nunique()),
        "total_reads": int(df["duplicate_count"].sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {
        "airr": str(airr_path),
        "hilary": str(hilary_path),
        "meta": str(meta_path),
        **meta,
    }
