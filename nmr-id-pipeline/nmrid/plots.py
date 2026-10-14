"""Thesis figures (matplotlib, PNG at 300 dpi; every figure also has a CSV of its numbers).

Colour follows the model in every figure: Random Forest blue, CNN orange,
Siamese aqua, cosine baseline yellow.  Lines also differ by marker shape and
carry a direct label at their end, so identity never depends on colour alone.
There is no dual-axis chart: loss and accuracy get their own panels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#ffffff"
MODEL_STYLE = {
    "rf": {"color": "#2a78d6", "marker": "o", "label": "Random Forest"},
    "cnn": {"color": "#eb6834", "marker": "s", "label": "1D CNN"},
    "siamese": {"color": "#1baf7a", "marker": "D", "label": "Siamese CNN"},
    "cosine": {"color": "#eda100", "marker": "^", "label": "Cosine search (no learning)"},
    "cnn_knn": {"color": "#eb6834", "marker": "s", "label": "CNN embedding + nearest neighbour"},
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _save(fig, path: str | Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _line(ax, x, y, key):
    st = MODEL_STYLE.get(key, {"color": SERIES[0], "marker": "o", "label": key})
    ax.plot(x, y, color=st["color"], marker=st["marker"], markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=st["label"], zorder=3)
    return (x[-1], y[-1]) if len(x) else None


def _end_labels(ax, ends, min_gap=0.055, fmt="{:.2f}"):
    """Value labels at the line ends, nudged apart vertically so they never overlap."""
    from matplotlib.transforms import offset_copy

    text_tf = offset_copy(ax.transData, fig=ax.figure, x=8, y=0, units="points")
    placed = None
    for x, y in sorted((e for e in ends if e is not None), key=lambda e: e[1]):
        ty = y if placed is None else max(y, placed + min_gap)
        placed = ty
        ax.annotate(fmt.format(y), xy=(x, y), xytext=(x, ty), textcoords=text_tf, va="center",
                    fontsize=8, color=TEXT_2, annotation_clip=False)


# ------------------------------------------------------------------ figures
def training_curves(hist: dict, path, title: str):
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    a1.plot(ep, hist["train_loss"], color=SERIES[0], label="training")
    if "val_loss" in hist:
        a1.plot(ep, hist["val_loss"], color=SERIES[1], label="validation")
    a1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    acc_key = "train_acc" if "train_acc" in hist else "train_pair_acc"
    a2.plot(ep, hist[acc_key], color=SERIES[0],
            label="training" if acc_key == "train_acc" else "training (pair accuracy)")
    val_key = "val_acc" if "val_acc" in hist else "val_top1"
    a2.plot(ep, hist[val_key], color=SERIES[1], label="validation (top-1)")
    a2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    for ax in (a1, a2):
        if len(ax.get_lines()) > 1:  # one series needs no legend; the panel title names it
            ax.legend(loc="best")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT)
    _save(fig, path)


def rq1_bars(df, path, metric_cols=("top1", "top5", "macro_f1"),
             metric_labels=("Top-1 accuracy", "Top-5 accuracy", "Macro F1")):
    _style()
    models = list(df["model"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    n = len(models)
    width = min(0.8 / n, 0.22)
    x = np.arange(len(metric_cols))
    for i, m in enumerate(models):
        st = MODEL_STYLE.get(m, {"color": SERIES[i % 4], "label": m})
        vals = [df.loc[df["model"] == m, c].values[0] for c in metric_cols]
        err = [df.loc[df["model"] == m, c + "_std"].values[0] if c + "_std" in df else 0 for c in metric_cols]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.9, color=st["color"], label=st["label"],
               yerr=err if any(e > 0 for e in err) else None, ecolor=TEXT_2, capsize=2, zorder=2)
    ax.set_xticks(x, metric_labels)
    ax.set(ylim=(0, 1.05), ylabel="Score", title="RQ1 - closed-set identification on the test replicates")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(n, 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def rq2_curves(df, x_col, path, xlabel, title):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ends = [_line(ax, *(lambda d: (d[x_col].values, d["top1"].values))(
        df[df["model"] == m].sort_values(x_col)), m) for m in df["model"].unique()]
    _end_labels(ax, ends)
    ax.set(xlabel=xlabel, ylabel="Top-1 accuracy", ylim=(0, 1.05), title=title)
    ax.legend(loc="lower left")
    _save(fig, path)


def rq3_bars(df, path):
    _style()
    df = df.sort_values("top1")
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(df) + 1.2))
    ax.barh(df["label"], df["top1"], height=0.55, color=SERIES[0], zorder=2)
    for yv, v in enumerate(df["top1"]):
        ax.annotate(f"{v:.3f}", (v, yv), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=TEXT_2)
    ax.set(xlim=(0, 1.1), xlabel="Top-1 accuracy", title="RQ3 - hand-crafted features vs learned features")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def rq4_curves(df, path):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ends = []
    for m in df["model"].unique():
        sub = df[df["model"] == m].sort_values("train_spectra_per_compound")
        ends.append(_line(ax, sub["train_spectra_per_compound"].values, sub["top1"].values, m))
    _end_labels(ax, ends)
    ax.set(xlabel="Training spectra per compound", ylabel="Top-1 accuracy", ylim=(0, 1.05),
           title="RQ4 - learning curves")
    ax.legend(loc="lower right")
    _save(fig, path)


def confusion_heatmap(cm, labels, path, title):
    _style()
    k = len(labels)
    fig, ax = plt.subplots(figsize=(0.28 * k + 2.5, 0.28 * k + 2))
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    im = ax.imshow(cm, cmap=cmap, vmin=0)
    short = [str(s)[:22] for s in labels]
    ax.set_xticks(range(k), short, rotation=90, fontsize=6.5)
    ax.set_yticks(range(k), short, fontsize=6.5)
    ax.set(xlabel="Predicted compound", ylabel="True compound", title=title)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Test spectra", color=TEXT_2)
    cb.locator = matplotlib.ticker.MaxNLocator(integer=True)
    cb.update_ticks()
    cb.outline.set_visible(False)
    _save(fig, path)


def feature_importance_groups(groups: dict, path):
    _style()
    items = sorted(groups.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(items) + 1.2))
    ax.barh([k.replace("_", " ") for k, _ in items], [v for _, v in items], height=0.55,
            color=SERIES[0], zorder=2)
    for yv, (_, v) in enumerate(items):
        ax.annotate(f"{v:.2f}", (v, yv), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=TEXT_2)
    ax.set(xlabel="Share of total impurity decrease", title="Random Forest - importance by feature group",
           xlim=(0, max(v for _, v in items) * 1.2))
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def example_spectra(grid, series: list[tuple[str, np.ndarray]], path, title):
    """Overlay of e.g. reference, a simulated replicate and a real re-measurement."""
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    c = grid.centers
    for i, (label, spec) in enumerate(series):
        ax.plot(c, spec + 1.1 * (len(series) - 1 - i), color=SERIES[i % 4], linewidth=1.0, label=label)
    ax.invert_xaxis()
    ax.set(xlabel="Chemical shift (ppm)", yticks=[], title=title)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(series))
    _save(fig, path)
