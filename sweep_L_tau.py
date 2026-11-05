"""Run: python3 sweep_L_tau.py --workers 10. Requires only NumPy.

Save log-spaced time-window means for 10 independent runs per (L, tau),
one CSV per parameter pair in data/paper_params/sweep_L_tau/. Plot separately with
python3 plot_sweep_L_tau.py. Motion settings come from two_disks.py.
Rerunning resumes completed samples; keep the same simulation settings.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import two_disks as simulation
from sweep_storage import HEADER, RAW_HEADER, compact_file, compact_sample, completed_means


L_VALUES = [2, 3, 4, 5, 6, 8, 10]
TAU_VALUES = np.arange(0, 3.01, 0.25)
SAMPLES = 10
DATA_DIR = Path(__file__).resolve().parent / "data" / "paper_params" / "sweep_L_tau"


def file_is_complete(path, length, tau, seed, first_step, stride, last_step):
    """Validate large completed CSVs in chunks; leave partial-file recovery below."""
    rows_per_sample = (last_step - first_step) // stride + 1
    with path.open("rb") as file:
        header = file.readline()
        if header.decode().strip().split(",") != RAW_HEADER:
            return False
        file.seek(max(0, path.stat().st_size - 1024))
        tail = file.read()
        if not tail.endswith(b"\n"):
            return False
        try:
            last = [float(value) for value in tail.splitlines()[-1].split(b",")]
            if last[:5] != [length, tau, SAMPLES, seed + SAMPLES - 1,
                            first_step + (rows_per_sample - 1) * stride]:
                return False
            file.seek(len(header))
            read = 0
            while True:
                position = file.tell()
                if not file.readline():
                    break
                file.seek(position)
                data = np.loadtxt(file, delimiter=",", max_rows=250_000, ndmin=2)
                if data.shape[1] != len(RAW_HEADER):
                    return False
                index = np.arange(read, read + len(data))
                samples = index // rows_per_sample + 1
                steps = first_step + (index % rows_per_sample) * stride
                if (not np.isfinite(data).all() or samples[-1] > SAMPLES
                        or not (data[:, 0] == length).all() or not (data[:, 1] == tau).all()
                        or not np.array_equal(data[:, 2], samples)
                        or not np.array_equal(data[:, 3], seed + samples - 1)
                        or not np.array_equal(data[:, 4], steps)
                        or not np.isclose(data[:, 5], (steps - 1) * simulation.DT).all()
                        or not np.isclose(data[:, 7], data[:, 6] / length).all()):
                    return False
                read += len(data)
            return read == SAMPLES * rows_per_sample
        except (ValueError, UnicodeDecodeError):
            return False


def _completed_raw_samples(path, length, tau, seed, first_step, stride, last_step):
    """Keep complete sample blocks; discard only an interrupted trailing block."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    if file_is_complete(path, length, tau, seed, first_step, stride, last_step):
        return SAMPLES
    rows_per_sample = (last_step - first_step) // stride + 1
    completed, count = 0, 0
    with path.open("rb+") as file:
        if file.readline().decode().strip().split(",") != RAW_HEADER:
            raise ValueError(f"Unexpected CSV header in {path}")
        keep = file.tell()
        end = path.stat().st_size
        while line := file.readline():
            # A partial final line is an interrupted write.
            if not line.endswith(b"\n"):
                break
            try:
                row = [float(value) for value in line.decode().strip().split(",")]
                if len(row) != len(RAW_HEADER):
                    raise ValueError("incomplete row")
            except (ValueError, UnicodeDecodeError):
                if file.tell() != end:
                    raise ValueError(f"Invalid CSV row in {path}")
                break
            step = first_step + count * stride
            if (row[:5] != [length, tau, completed + 1, seed + completed, step]
                    or not np.isclose(row[5], (step - 1) * simulation.DT)
                    or not np.isfinite(row).all()
                    or not np.isclose(row[7], row[6] / length)
                    or completed >= SAMPLES):
                raise ValueError(f"{path} does not match current simulation settings")
            count += 1
            if count == rows_per_sample:
                completed += 1
                count = 0
                keep = file.tell()
        if keep != end:
            file.truncate(keep)
    return completed


def completed_samples(path, length, tau, seed, first_step, stride, last_step):
    if path.exists() and path.stat().st_size:
        with path.open() as file:
            header = file.readline().strip().split(",")
        if header == RAW_HEADER:
            expected_count = (last_step - first_step) // stride + 1
            try:
                compact_file(path, expected_count=expected_count)
            except ValueError:
                # Recover an interrupted legacy run before compacting it.
                _completed_raw_samples(path, length, tau, seed, first_step, stride, last_step)
                compact_file(path, expected_count=expected_count)
    return completed_means(path, length, tau, seed, first_step, stride,
                           last_step, SAMPLES, simulation.DT)


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
    stride = simulation.STEP_SEPARATION
    first_step = ((max(1, simulation.STEADY_STATE_STEPS) + stride - 1) // stride) * stride
    rows = compact_sample(separations, length, first_step, stride, simulation.DT)
    return sample + 1, seed, rows


def run_sweep(workers=10):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_seed = simulation.SEED
    settings = {name: value for name, value in vars(simulation).items()
                if name.isupper() and isinstance(value, (int, float))}
    # Separation is measured before the position update at each sampling step.
    first_step = max(1, simulation.STEADY_STATE_STEPS)
    stride = simulation.STEP_SEPARATION
    first_step = ((first_step + stride - 1) // stride) * stride
    last_step = round(simulation.DURATION / simulation.DT) + simulation.STEADY_STATE_STEPS
    if first_step > last_step:
        raise ValueError("No separation sampling steps in this simulation")
    with ProcessPoolExecutor(max_workers=min(workers, SAMPLES),
                             mp_context=get_context("spawn")) as pool:
        for length in L_VALUES:
            for tau in TAU_VALUES:
                path = DATA_DIR / f"L{length:g}_tau{tau:g}.csv"
                done = completed_samples(path, length, tau, base_seed,
                                         first_step, stride, last_step)
                if done == SAMPLES:
                    print(f"Skipping L={length}, tau={tau:g}: already complete", flush=True)
                    continue
                print(f"L={length}, tau={tau:g}: resuming with {done}/{SAMPLES} saved", flush=True)
                # Independent samples; reuse seeds across parameter values.
                tasks = [(length, tau, sample, base_seed + sample, settings)
                         for sample in range(done, SAMPLES)]
                # Only the parent writes; counts retain each window's statistical weight.
                new_file = not path.exists() or path.stat().st_size == 0
                with path.open("a", newline="") as file:
                    writer = csv.writer(file)
                    if new_file:
                        writer.writerow(HEADER)
                    for sample, seed, rows in pool.map(run_sample, tasks):
                        writer.writerows([[length, tau, sample, seed, *row] for row in rows])
                        file.flush()
                print(f"Finished L={length}, tau={tau:g}: {SAMPLES} samples", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel sample processes on one node (default: 10)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be at least 1")
    run_sweep(args.workers)
    print(f"Saved log-spaced time-window means in {DATA_DIR}")
