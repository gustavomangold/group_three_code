"""Compare burn-in cutoffs on one long trajectory: python3 check_steady_state.py.

Each mean uses all samples after its cutoff, up to the same final step.
This checks sensitivity to burn-in; one trajectory cannot prove equilibration.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import two_disks as simulation


CUTOFFS = list(range(1_000_000, 10_000_001, 1_000_000))
TOTAL_STEPS = 11_000_000


def main():
    simulation.DURATION = TOTAL_STEPS * simulation.DT
    simulation.STEADY_STATE_STEPS = 0
    simulation.SAVE_EVERY = TOTAL_STEPS  # No animation frames needed for this check.
    print(f"Running {TOTAL_STEPS:,} steps: L={simulation.ARENA_DIAMETER:g}, "
          f"tau={simulation.PERSISTENCE_TIME:g}, seed={simulation.SEED}", flush=True)
    separations = simulation.simulate()[3]
    sample_steps = np.arange(1, len(separations) + 1) * simulation.STEP_SEPARATION
    means = [separations[sample_steps >= cutoff].mean() for cutoff in CUTOFFS]

    data_dir = Path(__file__).resolve().parent / "data" / "paper_params"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "steady_state_check_1M_10M.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["steady_state_steps", "samples", "mean_separation"])
        for cutoff, mean in zip(CUTOFFS, means):
            writer.writerow([cutoff, np.count_nonzero(sample_steps >= cutoff), mean])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(CUTOFFS, means, "o-")
    ax.set(xlabel="Steady-state steps discarded", ylabel="Mean separation / disk diameter",
           title=f"L={simulation.ARENA_DIAMETER:g}, τ={simulation.PERSISTENCE_TIME:g}, "
                 f"seed={simulation.SEED}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(data_dir / "steady_state_check_1M_10M.png", dpi=150)
    plt.close(fig)
    print(f"Saved steady_state_check_1M_10M.csv and steady_state_check_1M_10M.png in {data_dir}")


if __name__ == "__main__":
    main()
