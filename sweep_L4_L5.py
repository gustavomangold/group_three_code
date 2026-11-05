"""Sweep L=4,5 and tau=0..3 in steps of 0.1; 50 independent runs per point.

Run: python3 sweep_L4_L5.py --workers 10
Uses two_disks.py settings and resumes saved samples automatically.
"""

import argparse

import numpy as np
import sweep_L_tau as sweep


sweep.L_VALUES = [4, 5]
sweep.TAU_VALUES = np.arange(31) / 10
sweep.SAMPLES = 50
sweep.DATA_DIR = sweep.DATA_DIR.parent / "sweep_L4_L5_tau01"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be at least 1")
    sweep.run_sweep(args.workers)
    print(f"Saved full separation series in {sweep.DATA_DIR}")
