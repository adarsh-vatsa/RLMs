# /// script
# requires-python = ">=3.13"
# dependencies = ["matplotlib"]
# ///
"""Draw the paper's figures from the suite results into figure1.pdf and figure2.pdf.

Run from anywhere: uv run aaai-2027/figures/figures.py
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SUITES = HERE.parents[1] / "benchmark_artifacts"
MRCR = SUITES / "mrcr_v2/suites/20261029"

PANELS = [
    ("AA-LCR, accuracy", SUITES / "aa_lcr/suites/20261029/results-gpt-5.6-luna.csv", "accuracy"),
    ("MRCR v2, exact match", MRCR / "results.csv", "exact match"),
    ("LongBench-v2, accuracy", SUITES / "longbench_v2/suites/20261029/results.csv", "accuracy"),
]
WINDOW_LABELS = {32768: "32K", 65536: "64K", 81920: "80K", 131072: "128K", 262144: "262K"}
CONVERSATIONS = ["133K", "267K", "533K", "1.06M"]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Base condition name: legend label, colour, marker, line style; drawn in this order, bottom to top.
STYLE = {
    "middle": ("Middle truncation", "#52514e", "s", "-"),
    "position": ("Position", "#1baf7a", "^", "-"),
    "bm25": ("BM25", "#eb6834", "D", "-"),
    "upgraded": ("Fusion recipe", "#2a78d6", "o", "-"),
    "oracle": ("Oracle", INK, "*", ":"),
}
FULL_STYLE = dict(color=INK, lw=0.8, ls=(0, (4, 2)), marker="P", ms=5.5, mec="white", mew=0.6)


def line_style(base):
    _, color, marker, ls = STYLE[base]
    return dict(color=color, ls=ls, lw=1.2, marker=marker, ms=6.5 if marker == "*" else 4.5, mec="white", mew=0.6)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def style_axes(ax):
    ax.set_ylim(0, 80)
    ax.set_yticks(range(0, 81, 20))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7, length=2.5, width=0.6)


def load(path, metric):
    """Return {base condition: {window: score %}} and {window: share of the prompt middle truncation keeps}."""
    scores, share = {}, {}
    for row in read_csv(path):
        base = re.sub(r"_\d+k$", "", row["condition"])
        window = int(row["window"])
        scores.setdefault(base, {})[window] = 100 * float(row[metric])
        if base == "middle":
            share[window] = float(row["prompt kept"])
    return scores, share


def figure1():
    """Score against the share of the prompt that fits the budget, one panel per benchmark."""
    fig = Figure(figsize=(7.0, 2.35), layout="constrained")
    axes = fig.subplots(1, len(PANELS), sharey=True)
    for ax, (title, path, metric) in zip(axes, PANELS):
        scores, share = load(path, metric)
        if "full" in scores:
            full = scores["full"][262144]
            ax.axhline(full, color=INK, lw=0.8, ls=(0, (4, 2)), zorder=1)
            ax.plot(1.0, full, zorder=7, clip_on=False, **{**FULL_STYLE, "ls": "none"})
        for z, base in enumerate(STYLE, start=2):
            if base in scores:
                windows = sorted(scores[base])
                ax.plot([share[w] for w in windows], [scores[base][w] for w in windows], zorder=z, clip_on=False,
                        **line_style(base))
        budgets = ax.secondary_xaxis("top")
        budgets.set_xticks([share[w] for w in sorted(share)], [WINDOW_LABELS[w] for w in sorted(share)])
        budgets.tick_params(length=0, pad=1.5, labelsize=6.5, labelcolor=MUTED)
        budgets.spines["top"].set_visible(False)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.set_xlim(0, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        style_axes(ax)
    axes[0].set_ylabel("Score (%)", fontsize=8)
    fig.supxlabel("Mean share of the prompt that fits the input budget; context window above each panel", fontsize=8)

    bases = list(reversed(STYLE))
    handles = [Line2D([], [], **line_style(base)) for base in bases] + [Line2D([], [], **FULL_STYLE)]
    labels = [STYLE[base][0] for base in bases] + ["Full context (262K)"]
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False, fontsize=7.5,
               handlelength=2.6, columnspacing=1.4)
    return fig


def figure2():
    """MRCR exact match by conversation length at the 262K window."""
    rows = {row["condition"]: row for row in read_csv(MRCR / "results.csv")}
    kept = defaultdict(list)
    for line in (MRCR / "direct/20261029_middle/predictions.jsonl").open():
        answer = json.loads(line)
        kept[answer["full_rendered_input_tokens"] // 10000].append(
            answer["final_rendered_input_tokens"] / answer["full_rendered_input_tokens"])
    shares = [sum(values) / len(values) for _, values in sorted(kept.items())]

    fig = Figure(figsize=(3.3, 2.15), layout="constrained")
    ax = fig.subplots()
    xs = range(len(CONVERSATIONS))
    bases = ["middle", "bm25", "upgraded", "oracle"]
    for z, base in enumerate(bases, start=2):
        ax.plot(xs, [100 * float(rows[base][f"exact match at {size}"]) for size in CONVERSATIONS], zorder=z,
                clip_on=False, **line_style(base))
    ax.set_xticks(xs, [f"{size}\n{share:.0%} fits" for size, share in zip(CONVERSATIONS, shares)])
    ax.set_xlim(-0.25, len(CONVERSATIONS) - 0.75)
    ax.set_xlabel("Conversation length at the 262K window", fontsize=8)
    ax.set_ylabel("Exact match (%)", fontsize=8)
    style_axes(ax)
    fig.legend([Line2D([], [], **line_style(base)) for base in reversed(bases)],
               [STYLE[base][0] for base in reversed(bases)], loc="outside upper center", ncol=2, frameon=False,
               fontsize=7.5, handlelength=2.6, columnspacing=1.4)
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    })
    figure1().savefig(HERE / "figure1.pdf", metadata={"CreationDate": None})
    figure2().savefig(HERE / "figure2.pdf", metadata={"CreationDate": None})
