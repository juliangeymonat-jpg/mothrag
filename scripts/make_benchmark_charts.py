#!/usr/bin/env python3
"""Generate the benchmark comparison charts as PNGs.

Reproducible, version-controlled source for the dark-theme bar charts shown
in the README and on the website. Numbers are the published F1 figures
(HotpotQA / 2WikiMultiHopQA / MuSiQue, Llama-3.3-70B reader, n=1000), with
competitor rows reproduced from HippoRAG2's published comparison.

Design: competitor bars muted gray, MOTHRAG bars in the dataset colours so it
pops; value labels above every bar; truncated y-axis matching the existing
`benchmark_comparison.png` style. MOTHRAG is placed rightmost (the punchline).

Usage:  python scripts/make_benchmark_charts.py
Output: assets/benchmark_popular.png  (MOTHRAG vs the most popular RAG systems)

The GPU-frontier parity chart (`assets/benchmark_comparison.png`,
MOTHRAG vs HippoRAG 2 / CoRAG / NeocorRAG) is intentionally NOT overwritten;
its data is kept here for reference / future regeneration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- palette (matches the site dark theme) ----
BG = "#0a0e17"
GRID = "#1b2436"
AXIS_TEXT = "#9fb0c6"
VALUE_TEXT = "#e8eef6"
GRAY = "#39414f"          # competitor bars
CAPTION = "#5b9bff"
COLORS = ["#1fe3d2", "#2f9bf0", "#3b5be0"]   # HotpotQA / 2Wiki / MuSiQue
DATASETS = ["HotpotQA", "2WikiMultiHopQA", "MuSiQue"]
HIGHLIGHT = "MOTHRAG"

# ---- data (F1, n=1000, Llama-3.3-70B; competitors repro from HippoRAG2 Tab.2) ----
POPULAR = [
    ("RAPTOR", [69.5, 52.1, 28.9]),
    ("GraphRAG", [68.6, 58.6, 38.5]),
    ("HippoRAG 2", [75.5, 71.0, 48.6]),
    ("MOTHRAG", [78.1, 76.3, 50.5]),
]

# Parity / GPU-frontier set, kept for reference (NOT rendered by default):
#   HippoRAG 2  75.5 / 71.0 / 48.6
#   CoRAG       75.1 / 75.1 / 52.9
#   NeocorRAG   78.3 / 76.1 / 52.6
#   MOTHRAG     78.1 / 76.3 / 50.5


def make_chart(rows, outfile: Path, subtitle: str) -> None:
    systems = [name for name, _ in rows]
    n = len(systems)
    width = 0.26
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for j, _ds in enumerate(DATASETS):
        off = (j - 1) * width
        for i, (name, vals) in enumerate(rows):
            v = vals[j]
            color = COLORS[j] if name == HIGHLIGHT else GRAY
            ax.bar(xs[i] + off, v, width * 0.9, color=color, zorder=3)
            ax.text(xs[i] + off, v + 0.7, f"{v:.1f}", ha="center", va="bottom",
                    color=VALUE_TEXT, fontsize=8.5, fontweight="bold", zorder=4)

    allv = [v for _, vals in rows for v in vals]
    ax.set_ylim(min(allv) - 6, max(allv) + 6)
    ax.set_xticks(xs)
    ax.set_xticklabels(systems, fontsize=11)
    for lbl, (name, _) in zip(ax.get_xticklabels(), rows):
        if name == HIGHLIGHT:
            lbl.set_fontweight("bold")
            lbl.set_color(VALUE_TEXT)
        else:
            lbl.set_color(AXIS_TEXT)

    ax.set_ylabel("F1 score", color=AXIS_TEXT, fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=AXIS_TEXT, labelsize=9, length=0)
    ax.tick_params(axis="x", colors=AXIS_TEXT, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(subtitle, loc="left", color=AXIS_TEXT, fontsize=11.5, pad=16)

    handles = [Patch(facecolor=COLORS[j], label=DATASETS[j]) for j in range(3)]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True,
                    fontsize=9, handlelength=1.0, borderpad=0.8)
    leg.get_frame().set_facecolor("#0e1422")
    leg.get_frame().set_edgecolor("#26324a")
    for txt in leg.get_texts():
        txt.set_color(AXIS_TEXT)

    fig.text(0.98, 0.02, "commodity APIs only · no GPU", ha="right",
             color=CAPTION, style="italic", fontsize=9.5)

    fig.savefig(outfile, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "assets" / "benchmark_popular.png"
    make_chart(POPULAR, out, "MOTHRAG against the most popular RAG systems")
