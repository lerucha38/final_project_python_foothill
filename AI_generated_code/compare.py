"""Compare clonal assignments across methods."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def assignments_to_frame(assignments: dict[str, str], method: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"sequence_id": list(assignments.keys()), "clone_id": list(assignments.values()), "method": method}
    )


def load_assignments(path: Path, method: str, id_col: str = "sequence_id", clone_col: str = "clone_id") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=[id_col, clone_col])
    df = df.rename(columns={id_col: "sequence_id", clone_col: "clone_id"})
    df["method"] = method
    return df


def pairwise_overlap_metrics(
    left: pd.Series,
    right: pd.Series,
) -> dict[str, float]:
    common_idx = left.index.intersection(right.index)
    a = left.loc[common_idx].astype(str)
    b = right.loc[common_idx].astype(str)

    return {
        "n_shared_sequences": int(len(common_idx)),
        "adjusted_rand_index": float(adjusted_rand_score(a, b)),
        "normalized_mutual_info": float(normalized_mutual_info_score(a, b)),
    }


def clone_size_summary(df: pd.DataFrame, weight_col: str | None = "duplicate_count") -> pd.DataFrame:
    if weight_col and weight_col in df.columns:
        sizes = df.groupby("clone_id")[weight_col].sum().rename("clone_size")
    else:
        sizes = df.groupby("clone_id").size().rename("clone_size")

    summary = sizes.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).to_frame().T
    summary["n_clones"] = sizes.shape[0]
    summary["n_singletons"] = int((sizes == 1).sum())
    summary["largest_clone"] = int(sizes.max()) if len(sizes) else 0
    return summary


def compare_all_methods(assignments: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap_rows = []
    size_rows = []

    for method, df in assignments.items():
        summary = clone_size_summary(df)
        summary.insert(0, "method", method)
        size_rows.append(summary)

    methods = list(assignments.keys())
    indexed = {m: df.set_index("sequence_id")["clone_id"] for m, df in assignments.items()}

    for m1, m2 in combinations(methods, 2):
        metrics = pairwise_overlap_metrics(indexed[m1], indexed[m2])
        metrics.update({"method_a": m1, "method_b": m2})
        overlap_rows.append(metrics)

    return pd.DataFrame(overlap_rows), pd.concat(size_rows, ignore_index=True)


def top_clones(df: pd.DataFrame, n: int = 10, weight_col: str | None = "duplicate_count") -> pd.DataFrame:
    if weight_col and weight_col in df.columns:
        sizes = df.groupby("clone_id")[weight_col].sum().sort_values(ascending=False)
    else:
        sizes = df.groupby("clone_id").size().sort_values(ascending=False)
    top = sizes.head(n).reset_index()
    top.columns = ["clone_id", "clone_size"]
    return top


def _top_clone_entries(df: pd.DataFrame, top_n: int = 10, weight_col: str | None = "duplicate_count") -> list[dict]:
    ranked = top_clones(df, n=top_n, weight_col=weight_col)
    entries: list[dict] = []
    for rank, row in enumerate(ranked.itertuples(), start=1):
        clone_id = row.clone_id
        seqs = frozenset(df.loc[df.clone_id == clone_id, "sequence_id"])
        entries.append(
            {
                "clone_id": str(clone_id),
                "rank": rank,
                "n_sequences": int(row.clone_size),
                "seqs": seqs,
            }
        )
    return entries


def _best_overlap_match(
    seed_seqs: frozenset[str],
    candidates: list[dict],
    min_overlap: float,
) -> dict | None:
    best: dict | None = None
    for candidate in candidates:
        intersection = len(seed_seqs & candidate["seqs"])
        if intersection == 0:
            continue
        overlap = intersection / min(len(seed_seqs), len(candidate["seqs"]))
        if overlap >= min_overlap and (best is None or intersection > best["intersection"]):
            best = {**candidate, "intersection": intersection, "overlap": overlap}
    return best


def find_top_clone_shared_families(
    assignments: dict[str, pd.DataFrame],
    methods: list[str] | None = None,
    top_n: int = 10,
    min_overlap: float = 0.5,
    min_core: int = 1,
    junction_aa: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Find clonal families present in every method's top-N clones.

    Each top clone from any method seeds a candidate family. For every other
    method, the top-N clone with the largest sequence overlap is chosen. Families
    are kept only when all methods match above ``min_overlap`` and the sequence
    intersection across all matched clones is at least ``min_core``.
    """
    method_list = methods or list(assignments.keys())
    top_by_method = {
        method: _top_clone_entries(assignments[method], top_n=top_n, weight_col=None)
        for method in method_list
    }

    families: dict[tuple[tuple[str, str], ...], dict] = {}
    for seed_method in method_list:
        for seed in top_by_method[seed_method]:
            matched = {seed_method: seed}
            valid = True
            for method in method_list:
                if method == seed_method:
                    continue
                match = _best_overlap_match(seed["seqs"], top_by_method[method], min_overlap)
                if match is None:
                    valid = False
                    break
                matched[method] = match
            if not valid:
                continue

            key = tuple(sorted((method, matched[method]["clone_id"]) for method in method_list))
            if key in families:
                continue

            core = seed["seqs"]
            for method in method_list:
                core = core & matched[method]["seqs"]
            if len(core) < min_core:
                continue

            modal_cdr3 = ""
            if junction_aa and core:
                cdr3_counts: dict[str, int] = {}
                for seq_id in core:
                    cdr3 = junction_aa.get(seq_id)
                    if cdr3:
                        cdr3_counts[cdr3] = cdr3_counts.get(cdr3, 0) + 1
                if cdr3_counts:
                    modal_cdr3 = max(cdr3_counts, key=cdr3_counts.get)

            families[key] = {
                "seed_method": seed_method,
                "n_core_sequences": len(core),
                "modal_cdr3": modal_cdr3,
                "matched": matched,
            }

    rows: list[dict] = []
    ranked_families = sorted(
        families.values(),
        key=lambda family: (family["n_core_sequences"], max(v["n_sequences"] for v in family["matched"].values())),
        reverse=True,
    )
    for family_id, family in enumerate(ranked_families, start=1):
        for method in method_list:
            match = family["matched"][method]
            rows.append(
                {
                    "family_id": family_id,
                    "n_core_sequences": family["n_core_sequences"],
                    "modal_cdr3": family["modal_cdr3"],
                    "seed_method": family["seed_method"],
                    "method": method,
                    "clone_id": match["clone_id"],
                    "n_sequences": match["n_sequences"],
                    "top10_rank": match["rank"],
                    "overlap_with_seed": match.get("overlap", 1.0),
                }
            )

    return pd.DataFrame(rows)


def count_top_clone_method_coverage(
    assignments: dict[str, pd.DataFrame],
    methods: list[str] | None = None,
    top_n: int = 10,
    min_overlap: float = 0.5,
) -> pd.DataFrame:
    """Count top-N clone overlap groups by how many methods each spans.

    Top clones from each method are linked when sequence overlap is at least
    ``min_overlap`` (relative to the smaller clone). Connected components are
    deduplicated and tallied by the number of distinct methods they touch.
    """
    from collections import defaultdict

    method_list = methods or list(assignments.keys())
    n_methods_total = len(method_list)

    nodes: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for method in method_list:
        for entry in _top_clone_entries(assignments[method], top_n=top_n, weight_col=None):
            key = (method, entry["clone_id"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            nodes.append({**entry, "method": method})

    if not nodes:
        return pd.DataFrame(columns=["n_methods_found", "n_top_clone_groups", "n_methods_total"])

    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in combinations(range(len(nodes)), 2):
        if nodes[i]["method"] == nodes[j]["method"]:
            continue
        intersection = len(nodes[i]["seqs"] & nodes[j]["seqs"])
        if intersection == 0:
            continue
        score = intersection / min(len(nodes[i]["seqs"]), len(nodes[j]["seqs"]))
        if score >= min_overlap:
            union(i, j)

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(nodes)):
        components[find(i)].append(i)

    bucket_counts: dict[int, int] = {}
    for comp_idxs in components.values():
        n_methods = len({nodes[i]["method"] for i in comp_idxs})
        bucket_counts[n_methods] = bucket_counts.get(n_methods, 0) + 1

    return pd.DataFrame(
        {
            "n_methods_found": range(1, n_methods_total + 1),
            "n_top_clone_groups": [bucket_counts.get(k, 0) for k in range(1, n_methods_total + 1)],
            "n_methods_total": n_methods_total,
        }
    )


def clone_composition_stats(df: pd.DataFrame) -> dict[str, float | int]:
    """Clone counts and sequence fractions for size-1 and size-2 lineages (by row count)."""
    sizes = df.groupby("clone_id").size()
    total_seqs = int(len(df))
    total_clones = int(len(sizes))
    if total_seqs == 0:
        return {
            "n_total_sequences": 0,
            "n_total_clones": 0,
            "n_clones_size_1": 0,
            "n_clones_size_2": 0,
            "n_sequences_in_size_1_clones": 0,
            "n_sequences_in_size_2_clones": 0,
            "pct_sequences_in_size_1_clones": 0.0,
            "pct_sequences_in_size_2_clones": 0.0,
        }

    mask1 = sizes == 1
    mask2 = sizes == 2
    seqs_in_1 = int(mask1.sum())
    seqs_in_2 = int(sizes[mask2].sum())

    return {
        "n_total_sequences": total_seqs,
        "n_total_clones": total_clones,
        "n_clones_size_1": int(mask1.sum()),
        "n_clones_size_2": int(mask2.sum()),
        "n_sequences_in_size_1_clones": seqs_in_1,
        "n_sequences_in_size_2_clones": seqs_in_2,
        "pct_sequences_in_size_1_clones": 100.0 * seqs_in_1 / total_seqs,
        "pct_sequences_in_size_2_clones": 100.0 * seqs_in_2 / total_seqs,
    }
