"""Run with: python3 sweep_L_tau.py --workers 10. Results go into data/.

Each point averages 10 independent runs. Each run contributes its mean
separation after STEADY_STATE_STEPS, using the settings in two_disks.py.
L is the arena diameter in disk diameters; L=1 uses the same soft wall.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from multiprocessing import get_context
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without a display on the cluster.
import matplotlib.pyplot as plt
import numpy as np
import two_disks as simulation


L_VALUES = range(1, 11)
TAU_VALUES = np.arange(0, 3.01, 0.25)
SAMPLES = 10
DATA_DIR = Path(__file__).resolve().parent / "data"


def run_sample(task):
    length, tau, sample, seed, settings = task
    # Each process has its own simulation parameters and random generator.
    for name, value in settings.items():
        setattr(simulation, name, value)
    simulation.ARENA_DIAMETER = length
    simulation.PERSISTENCE_TIME = tau
    simulation.SEED = seed
    separations = simulation.simulate()[3]
    if not separations.size:
        raise ValueError("No separations recorded; check duration and steady-state steps.")
    mean = separations.mean()
    return [length, tau, sample + 1, seed, mean, mean / length]


def run_sweep(workers=10):
    DATA_DIR.mkdir(exist_ok=True)
    base_seed = simulation.SEED
    settings = {name: value for name, value in vars(simulation).items()
                if name.isupper() and isinstance(value, (int, float))}
    with ProcessPoolExecutor(max_workers=min(workers, SAMPLES),
                             mp_context=get_context("spawn")) as pool, \
            (DATA_DIR / "sweep_samples.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["L", "tau", "sample", "seed", "mean_separation", "separation_over_L"])
        for length in L_VALUES:
            for tau in TAU_VALUES:
                # Independent samples; reuse seeds across parameter values.
                tasks = [(length, tau, sample, base_seed + sample, settings)
                         for sample in range(SAMPLES)]
                # Only the parent process writes the CSV.
                for row in pool.map(run_sample, tasks):
                    writer.writerow(row)
                    file.flush()
                print(f"Finished L={length}, tau={tau:g}: {SAMPLES} samples", flush=True)


def plot_results():
    # Read the saved CSV so plotting uses exactly the recorded measurements.
    groups = {}
    with (DATA_DIR / "sweep_samples.csv").open() as file:
        for row in csv.DictReader(file):
            key = (float(row["L"]), float(row["tau"]))
            groups.setdefault(key, []).append(float(row["separation_over_L"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    with (DATA_DIR / "sweep_means.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["L", "tau", "samples", "mean_separation_over_L"])
        for length in sorted({key[0] for key in groups}):
            taus = sorted(tau for size, tau in groups if size == length)
            means = []
            for tau in taus:
                values = groups[length, tau]
                means.append(np.mean(values))
                writer.writerow([length, tau, len(values), means[-1]])
            ax.plot(taus, means, "o-", markersize=3, label=f"L={length:g}")
    ax.set(xlabel="τ", ylabel="Mean separation / L")
    ax.legend(title="Arena diameter", ncol=2)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "sweep_L_tau.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel sample processes on one node (default: 10)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be at least 1")
    run_sweep(args.workers)
    plot_results()
    print(f"Saved CSV files and plot in {DATA_DIR}")
