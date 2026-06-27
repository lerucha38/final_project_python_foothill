"""Convert MiXCR and HILARY inputs into a unified AIRR-style table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _gene_prefix(value: str) -> str:
    if pd.isna(value) or value == "":
        return ""
    return str(value).split("*")[0]


def _translate_nt(seq: str) -> str:
    table = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }
    seq = str(seq).upper()
    aa = []
    for i in range(0, len(seq) - 2, 3):
        aa.append(table.get(seq[i : i + 3], "X"))
    return "".join(aa)


def load_mixcr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    out = pd.DataFrame(
        {
            "sequence_id": df["cloneId"].astype(str) + "_" + df.index.astype(str),
            "v_call": df["bestVHit"].astype(str),
            "j_call": df["bestJHit"].astype(str),
            "v_gene": df["bestVGene"].astype(str),
            "j_gene": df["bestJGene"].astype(str),
            "junction": df["nSeqCDR3"].astype(str),
            "junction_aa": df["aaSeqCDR3"].astype(str),
            "junction_length": df["nSeqCDR3"].astype(str).str.len(),
            "duplicate_count": pd.to_numeric(df["readCount"], errors="coerce").fillna(1).astype(int),
            "locus": "IGH",
        }
    )
    return _clean_airr(out)


def load_hilary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    out = pd.DataFrame(
        {
            "sequence_id": df["sequence_id"].astype(str),
            "v_call": df["v_call"].astype(str),
            "j_call": df["j_call"].astype(str),
            "v_gene": df["v_call"].map(_gene_prefix),
            "j_gene": df["j_call"].map(_gene_prefix),
            "junction": df["junction"].astype(str),
            "junction_aa": df["junction"].map(_translate_nt),
            "junction_length": df["junction"].astype(str).str.len(),
            "duplicate_count": 1,
            "locus": "IGH",
            "v_sequence_alignment": df.get("v_sequence_alignment"),
            "j_sequence_alignment": df.get("j_sequence_alignment"),
            "v_germline_alignment": df.get("v_germline_alignment"),
            "j_germline_alignment": df.get("j_germline_alignment"),
        }
    )
    return _clean_airr(out)


def _clean_airr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["junction"].notna() & (df["junction"] != "nan")]
    df = df[df["v_call"].notna() & (df["v_call"] != "nan")]
    df = df[df["j_call"].notna() & (df["j_call"] != "nan")]
    df = df[df["junction_length"] >= 3]
    df = df.reset_index(drop=True)
    return df


def to_hilary_table(df: pd.DataFrame, include_alignments: bool = False) -> pd.DataFrame:
    base = df[["sequence_id", "v_call", "j_call"]].copy()
    if include_alignments and {"v_sequence_alignment", "j_sequence_alignment"}.issubset(df.columns):
        base["junction"] = df["junction"]
        base["v_sequence_alignment"] = df["v_sequence_alignment"]
        base["j_sequence_alignment"] = df["j_sequence_alignment"]
        base["v_germline_alignment"] = df["v_germline_alignment"]
        base["j_germline_alignment"] = df["j_germline_alignment"]
        return base

    # Placeholder alignments keep HILARy preprocessing happy when SHM data are absent.
    base["cdr3"] = df["junction"]
    base["alt_sequence_alignment"] = "A"
    base["alt_germline_alignment"] = "A"
    return base


def load_dataset(source_path: Path, source_format: str) -> pd.DataFrame:
    if source_format == "mixcr":
        return load_mixcr(source_path)
    if source_format == "hilary":
        return load_hilary(source_path)
    raise ValueError(f"Unsupported source format: {source_format}")


def prepare_dataset(
    dataset_key: str,
    source_path: Path,
    source_format: str,
    prepared_dir: Path,
) -> dict[str, Path]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(source_path, source_format)

    airr_path = prepared_dir / f"{dataset_key}.airr.tsv"
    hilary_path = prepared_dir / f"{dataset_key}.hilary.tsv"

    df.to_csv(airr_path, sep="\t", index=False)

    include_alignments = source_format == "hilary" and {
        "v_sequence_alignment",
        "v_germline_alignment",
    }.issubset(df.columns)
    to_hilary_table(df, include_alignments=include_alignments).to_csv(
        hilary_path, sep="\t", index=False, na_rep=""
    )

    return {"airr": airr_path, "hilary": hilary_path, "n_sequences": len(df)}
