#!/usr/bin/env python3
"""Generate a slide-ready schematic of the HILARY-CDR3 algorithm."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "figures"


def _box(ax, xy, w, h, text, fc, ec="#333333", fontsize=11, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end, color="#444444"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=color,
            shrinkA=6,
            shrinkB=6,
        )
    )


def plot_hilary_cdr3_schematic(out_path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Title
    ax.text(
        7,
        7.05,
        "HILARY-CDR3: adaptive single-linkage clonal inference",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        7,
        6.55,
        "Per V gene · J gene · CDR3-length (VJl) class",
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
    )

    # Main pipeline (top row)
    boxes = [
        (0.35, 4.55, 2.0, 1.15, "Input\nBCR sequences\n(CDR3 + V/J)", "#E8F4FD"),
        (2.85, 4.55, 2.0, 1.15, "Group by\nV · J · CDR3 length", "#DFF5E8"),
        (5.35, 4.55, 2.35, 1.15, "Pairwise distances\nwithin each class\n$x = n_{\\mathrm{Hamming}} / l$", "#FFF3D6"),
        (8.15, 4.55, 2.35, 1.15, "Mixture model (EM)\nrelated + unrelated", "#FDE8E8"),
        (10.95, 4.55, 2.35, 1.15, "Adaptive threshold\n$t^*$ (e.g. 99% precision)", "#EDE7F6"),
    ]
    for x, y, w, h, txt, fc in boxes:
        _box(ax, (x, y), w, h, txt, fc, fontsize=10.5)

    centers_top = [1.35, 3.85, 6.52, 9.32, 12.12]
    for i in range(len(centers_top) - 1):
        _arrow(ax, (centers_top[i] + 0.95, 5.12), (centers_top[i + 1] - 0.95, 5.12))

    # Bottom row: clustering output
    _box(
        ax,
        (4.8, 2.35),
        4.4,
        1.2,
        "Single-linkage clustering\nlink if CDR3 Hamming $< l \\cdot t^*$\n(fast prefix-tree implementation)",
        "#E8F4FD",
        fontsize=11,
        weight="bold",
    )
    _box(ax, (10.2, 2.35), 2.8, 1.2, "Output\nclonal families", "#DFF5E8", fontsize=11, weight="bold")

    _arrow(ax, (12.12, 4.55), (7.0, 3.65))
    _arrow(ax, (9.2, 2.95), (10.2, 2.95))

    # Mixture model inset
    inset = ax.inset_axes([0.06, 0.06, 0.38, 0.28])
    x = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    null = [0.05, 0.12, 0.22, 0.35, 0.28, 0.15, 0.08, 0.04, 0.02]
    related = [0.01, 0.02, 0.05, 0.12, 0.22, 0.18, 0.10, 0.05, 0.02]
    mixture = [0.06, 0.10, 0.14, 0.20, 0.24, 0.16, 0.09, 0.05, 0.03]

    inset.fill_between(x, null, alpha=0.35, color="#E57373", label="Unrelated ($P_F$, soNNia null)")
    inset.fill_between(x, related, alpha=0.45, color="#64B5F6", label="Related ($P_T$, Poisson)")
    inset.plot(x, mixture, color="#333333", linewidth=2.2, label="Observed mixture")
    inset.axvline(1.35, color="#7E57C2", linestyle="--", linewidth=2, label=r"Threshold $t^*$")
    inset.set_xlim(0, 4)
    inset.set_ylim(0, 0.42)
    inset.set_xlabel("Normalized CDR3 distance  $x = n/l$", fontsize=9)
    inset.set_ylabel("Density", fontsize=9)
    inset.set_title("Within one VJl class", fontsize=10, weight="bold", pad=6)
    inset.tick_params(labelsize=8)
    inset.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    # Legend / key idea
    ax.text(
        7.2,
        1.05,
        "Key idea: threshold adapts to CDR3 length and repertoire clonality ($\\rho$) in each VJl class",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
        style="italic",
    )

    # Color key strip
    legend_items = [
        mpatches.Patch(facecolor="#FDE8E8", edgecolor="#333", label="Statistical model"),
        mpatches.Patch(facecolor="#E8F4FD", edgecolor="#333", label="Clustering"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, frameon=True)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "hilary_cdr3_schematic.png"
    plot_hilary_cdr3_schematic(out)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
