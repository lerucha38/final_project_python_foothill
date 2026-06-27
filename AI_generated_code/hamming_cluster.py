"""Hamming-distance single-linkage clustering within V/J/junction-length groups."""

from __future__ import annotations

import os
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def _group_keys(df: pd.DataFrame) -> pd.Series:
    return (
        df["v_gene"].astype(str)
        + "|"
        + df["j_gene"].astype(str)
        + "|"
        + df["junction_length"].astype(str)
    )


def _single_linkage_labels(sequences: list[str], max_distance: float) -> np.ndarray:
    n = len(sequences)
    if n == 1:
        return np.array([1], dtype=int)

    matrix = np.array([list(seq.upper().encode("ascii")) for seq in sequences], dtype=np.uint8)
    dist = pdist(matrix, metric="hamming")
    if dist.size == 0:
        return np.array([1], dtype=int)

    z = linkage(dist, method="single")
    return fcluster(z, t=max_distance, criterion="distance")


def _init_hamming_worker() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"


def _process_group(task: tuple[str, list, list[str], float]) -> tuple[str, list[tuple[int, int]]]:
    group, idx_list, seqs, max_distance = task
    labels = _single_linkage_labels(seqs, max_distance=max_distance)
    return group, list(zip(idx_list, labels))


def cluster_hamming(
    df: pd.DataFrame,
    similarity: float,
    sequence_col: str = "junction",
    clone_col: str = "clone_id",
    nproc: int = 1,
) -> tuple[pd.DataFrame, dict]:
    if not 0 < similarity < 1:
        raise ValueError("similarity must be between 0 and 1")

    max_distance = 1.0 - similarity
    work = df.copy()
    work["_group"] = _group_keys(work)

    clone_ids = np.empty(len(work), dtype=object)
    tasks = [
        (
            group,
            list(idx),
            work.loc[list(idx), sequence_col].astype(str).tolist(),
            max_distance,
        )
        for group, idx in work.groupby("_group").groups.items()
    ]

    start = time.perf_counter()
    if nproc > 1 and len(tasks) > 1:
        chunksize = max(1, len(tasks) // (nproc * 4))
        with ProcessPoolExecutor(
            max_workers=nproc, initializer=_init_hamming_worker
        ) as pool:
            results = pool.map(_process_group, tasks, chunksize=chunksize)
    else:
        results = map(_process_group, tasks)

    for group, pairs in results:
        for row_idx, label in pairs:
            clone_ids[row_idx] = f"{group}::{label}"

    elapsed = time.perf_counter() - start
    work[clone_col] = clone_ids
    work = work.drop(columns=["_group"])

    stats = {
        "runtime_sec": elapsed,
        "n_sequences": len(work),
        "n_clone_groups": int(work[clone_col].nunique()),
        "similarity_threshold": similarity,
        "nproc": nproc,
    }
    return work, stats


def run_all_hamming_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    nproc: int = 1,
) -> dict[str, tuple[pd.DataFrame, dict]]:
    results = {}
    for name, similarity in thresholds.items():
        clustered, stats = cluster_hamming(
            df, similarity=similarity, clone_col="clone_id", nproc=nproc
        )
        stats["method"] = name
        results[name] = (clustered, stats)
    return results
