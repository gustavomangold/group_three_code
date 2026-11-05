"""Plot recorded sweep CSVs: python3 plot_sweep_L_tau.py.

Each plotted point averages the time means of independent runs (--samples).
Run this locally after copying data/periodic/sweep_L_tau/ from the cluster.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data" / "periodic" / "sweep_L_tau"


def plot_results(data_dir=DATA_DIR, expected_samples=10):
    paths = sorted(data_dir.glob("L*_tau*.csv"))
    if not paths:
        raise ValueError(f"No sweep CSV files found in {data_dir}")
    groups = {}
    errors = {}
    for path in paths:
        # Read only L, tau, sample and normalized separation.
        data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 7), ndmin=2)
        samples = np.unique(data[:, 2])
        if len(samples) != expected_samples:
            raise ValueError(f"{path.name}: expected {expected_samples} samples, found {len(samples)}")
        length, tau = data[0, :2]
        means = [data[data[:, 2] == sample, 3].mean()
                 for sample in samples]
        groups[length, tau] = np.mean(means)
        errors[length, tau] = np.std(means, ddof=1) / np.sqrt(len(means))

    lengths = sorted({key[0] for key in groups})
    columns = min(4, len(lengths))
    rows = (len(lengths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.1 * rows),
                             sharex=True, sharey=True, squeeze=False)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.82, len(lengths)))
    with (data_dir.parent / "sweep_means.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["L", "tau", "samples", "mean_separation_over_L"])
        for ax, length, color in zip(axes.flat, lengths, colors):
            taus = sorted(tau for size, tau in groups if size == length)
            means = np.array([groups[length, tau] for tau in taus])
            sem = np.array([errors[length, tau] for tau in taus])
            for tau, mean in zip(taus, means):
                writer.writerow([length, tau, expected_samples, mean])
            ax.fill_between(taus, means - sem, means + sem, color=color, alpha=0.16,
                            linewidth=0)
            ax.plot(taus, means, "o-", color=color, markersize=4, linewidth=1.8,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.set_title(f"L = {length:g}", loc="left", fontweight="bold", pad=10)
            ax.set_xticks(np.arange(0, 3.01, 1))
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#cbd5e1")
            ax.tick_params(length=0, pad=6)
    for ax in list(axes.flat)[len(lengths):]:
        ax.set_visible(False)
    fig.suptitle("Separation versus persistence time", fontsize=16, fontweight="bold")
    fig.supxlabel("Persistence time, τ", fontsize=12, y=0.055)
    fig.supylabel("Mean separation / L", fontsize=12)
    fig.text(0.5, 0.01, f"Each point: mean of {expected_samples} runs · Shading: ±1 standard error · Shared axis limits",
             ha="center", fontsize=9, color="#64748b")
    fig.tight_layout(rect=(0.025, 0.09, 1, 0.97), h_pad=2, w_pad=1.5)
    output = data_dir.parent / "sweep_L_tau.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("samples must be at least 2")
    plot_results(args.data_dir, args.samples)
