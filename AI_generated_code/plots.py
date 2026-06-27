"""Plotting helpers for BCR benchmark results."""

from __future__ import annotations

from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHOD_LABELS = {
    "hamming_80": "Hamming 80%",
    "hamming_90": "Hamming 90%",
    "hamming_95": "Hamming 95%",
    "hilary": "HILARY cdr3nt",
    "scoper_hierarchical": "SCOPer hierarchical",
    "scoper_spectral": "SCOPer spectral",
    "fastbcr": "fastBCR",
}


def plot_runtime_boxplot(
    runtime_df: pd.DataFrame,
    out_path: Path,
    title: str = "Runtime difference between different clonal inference approaches",
) -> None:
    if runtime_df.empty:
        raise ValueError("No runtime records to plot.")

    df = runtime_df.copy()
    method_order = [m for m in METHOD_LABELS if m in df["method"].unique()]
    df["method"] = pd.Categorical(
        df["method"].map(METHOD_LABELS).fillna(df["method"]),
        categories=[METHOD_LABELS[m] for m in method_order],
        ordered=True,
    )

    n_min = float(df["n_sequences"].min())
    n_max = float(df["n_sequences"].max())

    fig, ax = plt.subplots(figsize=(15, 6))
    box_color = "#98df8a"
    dot_color = "#7b3294"
    sns.boxplot(
        data=df,
        x="method",
        y="runtime_sec",
        order=[METHOD_LABELS[m] for m in method_order],
        whis=(0, 100),
        color=box_color,
        linewidth=1.2,
        fliersize=0,
        ax=ax,
    )

    size_ref = [n_min, float(df["n_sequences"].median()), n_max]
    size_labels = [f"{int(v):,}" for v in size_ref]
    handles = []
    for size, label in zip(size_ref, size_labels):
        handles.append(
            ax.scatter(
                [],
                [],
                s=_sequence_marker_size(size, n_min, n_max),
                c=dot_color,
                alpha=0.85,
                label=label,
            )
        )

    scatter = ax.scatter(
        df["method"].cat.codes,
        df["runtime_sec"],
        s=df["n_sequences"].apply(lambda n: _sequence_marker_size(n, n_min, n_max)),
        c=dot_color,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.tick_params(axis="x", rotation=30)

    size_legend = ax.legend(
        handles,
        size_labels,
        title="Sequences",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=True,
        labelspacing=1.4,
        borderpad=1.0,
    )
    ax.add_artist(size_legend)
    ax.legend(
        [scatter],
        ["Individual run"],
        loc="upper left",
        bbox_to_anchor=(1.01, 0.55),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.35)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def plot_singleton_pct_boxplot(
    composition_df: pd.DataFrame,
    out_path: Path,
    title: str = "Fraction of sequences in singleton clones",
) -> None:
    if composition_df.empty:
        raise ValueError("No clone composition records to plot.")

    df = composition_df.copy()
    if "pct_sequences_in_size_1_clones" not in df.columns:
        raise ValueError("composition_df must include pct_sequences_in_size_1_clones")


    method_order = [m for m in METHOD_LABELS if m in df["method"].unique()]
    df["method"] = pd.Categorical(
        df["method"].map(METHOD_LABELS).fillna(df["method"]),
        categories=[METHOD_LABELS[m] for m in method_order],
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    box_color = "#98df8a"
    dot_color = "#ff7f0e"

    pct_col = "pct_sequences_in_size_1_clones"
    pct_min = float(df[pct_col].min())
    pct_max = float(df[pct_col].max())
    size_ref = [pct_min, float(df[pct_col].median()), pct_max]
    size_labels = [f"{v:.1f}%" for v in size_ref]

    sns.boxplot(
        data=df,
        x="method",
        y=pct_col,
        order=[METHOD_LABELS[m] for m in method_order],
        whis=(0, 100),
        color=box_color,
        linewidth=1.2,
        fliersize=0,
        ax=ax,
    )

    handles = [
        ax.scatter(
            [],
            [],
            s=_sequence_marker_size(size, pct_min, pct_max),
            c=dot_color,
            alpha=0.85,
            label=label,
        )
        for size, label in zip(size_ref, size_labels)
    ]
    scatter = ax.scatter(
        df["method"].cat.codes,
        df[pct_col],
        s=df[pct_col].apply(lambda v: _sequence_marker_size(v, pct_min, pct_max)),
        c=dot_color,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    size_legend = ax.legend(
        handles,
        size_labels,
        title="Singleton %",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=True,
        labelspacing=1.4,
        borderpad=1.0,
    )
    ax.add_artist(size_legend)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("% of sequences in singleton clones")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(bottom=-2)

    ax.legend(
        [scatter],
        ["Individual donor"],
        loc="upper left",
        bbox_to_anchor=(1.01, 0.55),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.35)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _sequence_marker_size(
    n_sequences: float,
    n_min: float,
    n_max: float,
    size_min: float = 30.0,
    size_max: float = 500.0,
) -> float:
    if n_sequences <= 0:
        return size_min
    if n_max <= n_min:
        return (size_min + size_max) / 2.0
    frac = (n_sequences - n_min) / (n_max - n_min)
    return float(size_min + frac * (size_max - size_min))


def plot_runtime(runtime_df: pd.DataFrame, out_path: Path, title: str = "Runtime comparison") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=runtime_df, x="method", y="runtime_sec", ax=ax, hue="method", legend=False)
    ax.set_title(title)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_clone_size_distribution(
    assignments: pd.DataFrame,
    out_path: Path,
    weight_col: str = "duplicate_count",
    title: str = "Clone size distribution",
) -> None:
    if weight_col in assignments.columns:
        sizes = assignments.groupby("clone_id")[weight_col].sum()
    else:
        sizes = assignments.groupby("clone_id").size()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(sizes, bins=50, log_scale=(False, True), ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Clone size")
    ax.set_ylabel("Count (log scale)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overlap_heatmap(overlap_df: pd.DataFrame, out_path: Path, metric: str = "adjusted_rand_index") -> None:
    methods = sorted(set(overlap_df["method_a"]).union(set(overlap_df["method_b"])))
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for _, row in overlap_df.iterrows():
        matrix.loc[row["method_a"], row["method_b"]] = row[metric]
        matrix.loc[row["method_b"], row["method_a"]] = row[metric]
    for m in methods:
        matrix.loc[m, m] = 1.0

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(metric.replace("_", " ").title())
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cdr3_logo(
    sequences: list[str],
    out_path: Path,
    title: str = "CDR3 logo",
) -> None:
    if not sequences:
        return

    max_len = max(len(s) for s in sequences)
    counts = pd.DataFrame(0.0, index=list("ACDEFGHIKLMNPQRSTVWY"), columns=range(max_len))
    for seq in sequences:
        for i, aa in enumerate(seq.upper()):
            if aa in counts.index:
                counts.iat[counts.index.get_loc(aa), i] += 1

    counts = counts.T
    counts = counts.loc[:, counts.sum(axis=0) > 0]

    fig, ax = plt.subplots(figsize=(max(6, max_len * 0.35), 3))
    logomaker.Logo(counts, ax=ax, color_scheme="chemistry")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_clone_family_sizes(families_df: pd.DataFrame, out_path: Path) -> None:
    if families_df.empty:
        raise ValueError("No shared top-clone families to plot.")

    df = families_df.copy()
    method_order = [m for m in METHOD_LABELS if m in df["method"].unique()]
    df["method_label"] = pd.Categorical(
        df["method"].map(METHOD_LABELS).fillna(df["method"]),
        categories=[METHOD_LABELS[m] for m in method_order],
        ordered=True,
    )
    family_order = (
        df.drop_duplicates(["donor_id", "family_id"])
        .sort_values(["donor_id", "family_id"])
        .apply(
            lambda row: (
                f"{row['donor_id'].split('-')[-1]} F{row['family_id']} "
                f"(core={int(row['n_core_sequences']):,})"
            ),
            axis=1,
        )
        .tolist()
    )
    df["family_label"] = df.apply(
        lambda row: (
            f"{row['donor_id'].split('-')[-1]} F{row['family_id']} "
            f"(core={int(row['n_core_sequences']):,})"
        ),
        axis=1,
    )
    df["family_label"] = pd.Categorical(df["family_label"], categories=family_order, ordered=True)

    n_families = df[["donor_id", "family_id"]].drop_duplicates().shape[0]
    fig_h = max(6, 0.45 * n_families + 2)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.barplot(
        data=df,
        y="family_label",
        x="n_sequences",
        hue="method_label",
        order=family_order,
        hue_order=[METHOD_LABELS[m] for m in method_order],
        ax=ax,
    )
    ax.set_xlabel("Sequences assigned to matched top-10 clone")
    ax.set_ylabel("")
    ax.set_title("Shared top-10 clone families across all methods", fontsize=14, fontweight="bold")
    ax.legend(title="Method", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_ram_lineplot(ram_df: pd.DataFrame, out_path: Path) -> None:
    if ram_df.empty:
        raise ValueError("No RAM records to plot.")

    df = ram_df.copy()
    df = df[df["peak_rss_mb"] > 0]
    if df.empty:
        raise ValueError("No non-zero peak RAM records to plot.")

    if "peak_rss_gb" in df.columns:
        df["peak_rss_plot"] = df["peak_rss_gb"]
    else:
        df["peak_rss_plot"] = df["peak_rss_mb"] / 1024.0

    method_order = [m for m in METHOD_LABELS if m in df["method"].unique()]
    df["method_label"] = pd.Categorical(
        df["method"].map(METHOD_LABELS).fillna(df["method"]),
        categories=[METHOD_LABELS[m] for m in method_order],
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("tab10", n_colors=len(method_order))
    for color, method in zip(palette, method_order):
        subset = (
            df[df["method"] == method]
            .sort_values("n_sequences")
            .drop_duplicates(subset=["n_sequences"], keep="last")
        )
        ax.plot(
            subset["n_sequences"],
            subset["peak_rss_plot"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=METHOD_LABELS[method],
            color=color,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size (sequences, log scale)")
    ax.set_ylabel("Peak RAM (GB, log scale)")
    ax.set_title("Peak memory usage vs input size", fontsize=14, fontweight="bold")
    ax.legend(title="Method", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_top_clone_method_coverage(coverage_df: pd.DataFrame, out_path: Path) -> None:
    if coverage_df.empty:
        raise ValueError("No top-clone method coverage records to plot.")

    df = coverage_df.copy()
    df = df.drop_duplicates(subset=["donor_id", "n_methods_found"], keep="last")
    df["donor_label"] = df["donor_id"].str.replace("BFI-", "", regex=False)

    n_methods_total = int(df["n_methods_total"].max())
    donor_order = sorted(df["donor_label"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=df,
        x="n_methods_found",
        y="n_top_clone_groups",
        hue="donor_label",
        hue_order=donor_order,
        ax=ax,
        s=90,
        edgecolor="white",
        linewidth=0.6,
        palette="tab10",
    )
    ax.set_xticks(range(1, n_methods_total + 1))
    ax.set_xlabel("Found in how many methods (of top-10 clones)")
    ax.set_ylabel("Number of matched top-clone groups per donor")
    ax.set_title(
        "Top-10 clone overlap across methods by method coverage",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.legend(title="Donor", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
