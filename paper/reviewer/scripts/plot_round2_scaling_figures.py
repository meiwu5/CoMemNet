#!/usr/bin/env python3
"""Render the fixed-50-epoch PEMSD4(L) scaling figures for Reviewer 2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "figures"

BLUE = "#82B0D2"
GREEN = "#8ECFC9"
RED = "#FBC4BC"
INK = "#222222"
GRID = "#D9D9D9"

SCALES = ["25%", "50%", "75%", "100%"]
NODES = np.array([602, 1203, 1804, 2406])

COMEM_TIME = np.array([218.92, 272.28, 298.60, 361.58])
CURRENT_TIME = np.array([254.82, 391.34, 558.16, 618.32])
STID_TIME = np.array([412.33, 686.72, 769.39, 795.81])

COMEM_MAE = np.array([25.56, 23.76, 22.03, 21.99])
CURRENT_MAE = np.array([25.15, 24.21, 22.60, 22.69])
STID_MAE = np.array([25.27, 24.70, 23.13, 23.22])


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.edgecolor": INK,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def bold_ticks(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def finish(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(OUT / f"{name}.png", dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_relative_advantage():
    fig, (ax_mae, ax_time) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    x = np.arange(len(SCALES))
    mae_gain_current = (CURRENT_MAE - COMEM_MAE) / CURRENT_MAE * 100.0
    mae_gain_stid = (STID_MAE - COMEM_MAE) / STID_MAE * 100.0
    time_saved_current = (CURRENT_TIME - COMEM_TIME) / CURRENT_TIME * 100.0
    time_saved_stid = (STID_TIME - COMEM_TIME) / STID_TIME * 100.0

    mae_sets = [
        (-0.16, mae_gain_current, GREEN, "vs. Current-period"),
        (0.16, mae_gain_stid, RED, "vs. STID-current"),
    ]
    for offset, values, color, label in mae_sets:
        bars = ax_mae.bar(x + offset, values, 0.32, color=color, edgecolor=INK, linewidth=0.6, label=label)
        for bar, value in zip(bars, values):
            ax_mae.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.22 if value >= 0 else -0.32),
                f"{value:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )
    ax_mae.axhline(0, color=INK, linewidth=0.9)
    ax_mae.set_xlabel("PEMSD4(L) scale")
    ax_mae.set_ylabel("12-step MAE reduction of CoMemNet (%)")
    ax_mae.set_xticks(x, SCALES)
    ax_mae.set_ylim(-2.5, 6.8)
    ax_mae.set_yticks(np.arange(-2, 7, 2))
    ax_mae.grid(axis="y", color=GRID, linewidth=0.75)
    ax_mae.set_axisbelow(True)
    ax_mae.set_title("(a) Accuracy improvement", loc="left", pad=10)
    legend = ax_mae.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    time_sets = [
        (-0.16, time_saved_current, GREEN, "vs. Current-period"),
        (0.16, time_saved_stid, RED, "vs. STID-current"),
    ]
    for offset, values, color, label in time_sets:
        bars = ax_time.bar(x + offset, values, 0.32, color=color, edgecolor=INK, linewidth=0.6, label=label)
        for bar, value in zip(bars, values):
            ax_time.text(
                bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )
    ax_time.set_xlabel("PEMSD4(L) scale")
    ax_time.set_ylabel("Cumulative training time saved (%)")
    ax_time.set_xticks(x, SCALES)
    ax_time.set_ylim(0, 72)
    ax_time.set_yticks(np.arange(0, 71, 10))
    ax_time.grid(axis="y", color=GRID, linewidth=0.75)
    ax_time.set_axisbelow(True)
    ax_time.set_title("(b) Efficiency improvement", loc="left", pad=10)
    legend = ax_time.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    for ax in (ax_mae, ax_time):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        bold_ticks(ax)
    finish(fig, "round2_scaling_relative_advantage")


if __name__ == "__main__":
    set_style()
    plot_relative_advantage()
    print(f"Wrote figures to {OUT}")
