"""Two active disks in a periodic square, inspired by cell_simulation.

Run: python3 two_disks.py --L 6 --PERSISTENCE_TIME 2 (requires numpy and matplotlib).
Lengths below are in disk diameters: D=10 physically, square side=L*D.
Time and motion parameters are illustrative, not fitted to experiments.

Both disks propel themselves and diffuse. On contact, both turn away;
the green disk temporarily reduces its angular diffusion (more persistence).
Disk repulsion is a soft spring: force = REPULSION * overlap.
Positions wrap across opposite edges; interactions use the nearest image.
"""

import argparse
from pathlib import Path

import numpy as np


# Edit these parameters to explore the model. Mobility is set to one.
DISK_DIAMETER = 10.0      # Physical length represented by one unit
BOX_SIZE = 5.0                      # Square side L, in disk diameters
SPEED = 1.0                         # Disk diameters per time unit
REPULSION = 30.0                   # Spring strength / mobility
#TRANSLATIONAL_DIFFUSION = 0.02      # Disk diameters squared per time unit
ANGULAR_DIFFUSION = 10             # Radians squared per time unit
 
#PERSISTENT_DIFFUSION = 0.01         # Green disk after contact
PERSISTENCE_TIME = 2.0
DT = 0.001

STEP_SEPARATION = 5
STEADY_STATE_STEPS = 1000000

DURATION = 100.0
SAVE_EVERY = 20                     # One animation frame per 500 steps
SEED = 7
MU = 1


def repulsive_force(overlap, normal):
    """Soft repulsion between overlapping disks."""
    return REPULSION * max(0.0, overlap) * normal


def simulate():
    """Return times, positions (frame, disk, xy), and green persistence flags."""
    rng = np.random.default_rng(SEED)
    # Disk 0 is green, disk 1 is red; initially they face each other.
    offset = min(1.0, BOX_SIZE / 4)  # Avoid coincident periodic images for L=2.
    positions = np.array([[-offset, 0.0], [offset, 0.0]])
    angles = np.array([0.0, np.pi])
    persistent_left = 0.0
    times, tracks, persistent = [0.0], [positions.copy()], [False]

    separations = []

    for step in range(1, round(DURATION / DT) + STEADY_STATE_STEPS + 1):
        persistent_left = max(0.0, persistent_left - DT)
        separation = positions[0] - positions[1]
        separation -= BOX_SIZE * np.rint(separation / BOX_SIZE)  # Nearest image
        distance = np.linalg.norm(separation)
        # A fallback direction avoids division by zero at coincident centers.
        normal = separation / distance if distance > 1e-12 else np.array([1., 0.])
        forces = np.zeros((2, 2))

        if distance < 1.0:  # Contact: one disk diameter between centers
            forces[0] = repulsive_force(1.0 - distance, normal)
            forces[1] = -forces[0]
            away = np.arctan2(normal[1], normal[0])
            angles[:] = [away, away + np.pi]
            # Like the repo, contacts do not refresh an active persistence timer.
            if persistent_left == 0.0:
                persistent_left = PERSISTENCE_TIME

        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        # Euler-Maruyama: drift * dt + sqrt(2 * diffusion * dt) * normal noise.
        positions += (SPEED * directions + MU * forces) * DT
        #positions += np.sqrt(2 * TRANSLATIONAL_DIFFUSION * DT) * rng.normal(size=(2, 2))

        # Keep disk centers in [-L/2, L/2) on both axes.
        positions = (positions + BOX_SIZE / 2) % BOX_SIZE - BOX_SIZE / 2

        angular_diffusion_green = ANGULAR_DIFFUSION
        if persistent_left > 0:
            angular_diffusion_green = 0.0

        angular_diffusion = np.array([
            angular_diffusion_green,
            ANGULAR_DIFFUSION
        ])
        angles += np.sqrt(2 * angular_diffusion * DT) * rng.normal(size=2)

        if step >= STEADY_STATE_STEPS:
            if step % STEP_SEPARATION == 0:
                separations.append(distance)

        if step % SAVE_EVERY == 0 or step == round(DURATION / DT):
            times.append(step * DT)
            tracks.append(positions.copy())
            persistent.append(persistent_left > 0)

    return np.array(times), np.array(tracks), np.array(persistent), np.array(separations)


def animate(times, tracks, persistent, separations):
    """Show the arena; purple means the green disk has increased persistence."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(6, 6))
    half = BOX_SIZE / 2
    ax.add_patch(Rectangle((-half, -half), BOX_SIZE, BOX_SIZE, fill=False, color="black"))
    ax.set(xlim=(-half, half), ylim=(-half, half),
           xlabel="x / disk diameter", ylabel="y / disk diameter", aspect="equal")
    disks = []
    # Draw periodic copies so a disk crossing an edge appears on both sides.
    for i, color in enumerate(["tab:green", "tab:red"]):
        for dx in (-BOX_SIZE, 0, BOX_SIZE):
            for dy in (-BOX_SIZE, 0, BOX_SIZE):
                shift = np.array([dx, dy])
                disk = Circle(tracks[0, i] + shift, .5, color=color, alpha=.8)
                ax.add_patch(disk)
                disks.append((i, shift, disk))
    title = ax.set_title("")

    def update(frame):
        for i, shift, disk in disks:
            disk.center = tracks[frame, i] + shift
            disk.set_color(("tab:purple" if persistent[frame] else "tab:green")
                           if i == 0 else "tab:red")
        title.set_text(f"t = {times[frame]:.2f} | periodic square, L = {BOX_SIZE:g}D")
        return *[disk for i, shift, disk in disks], title
    
    animation = FuncAnimation(fig, update, frames=len(times), interval=40, blit=True)
    plt.show()
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=float, default=BOX_SIZE,
                        help="Square side in disk diameters (default: %(default)s)")
    parser.add_argument("--PERSISTENCE_TIME", "--persistence-time", type=float,
                        default=PERSISTENCE_TIME,
                        help="Persistence duration after contact (default: %(default)s)")
    parser.add_argument("--ANIMATE", "--animate", type=bool,
                            default=False,
                            help="Display animation (default: %(default)s)")
    parser.add_argument("--STEADY_STATE_STEPS", "--steady-state-steps", type=int,
                        default=STEADY_STATE_STEPS,
                        help="Steps discarded before sampling (default: %(default)s)")
    parser.add_argument("--DURATION", "--duration", type=float, default=DURATION,
                        help="Simulation duration (default: %(default)s)")
    args = parser.parse_args()
    if not np.isfinite(args.L) or args.L <= 1:
        parser.error("L must be finite and greater than 1 disk diameter")
    if not np.isfinite(args.PERSISTENCE_TIME) or args.PERSISTENCE_TIME < 0:
        parser.error("PERSISTENCE_TIME must be finite and nonnegative")
    if not np.isfinite(args.DURATION) or args.DURATION <= 0:
        parser.error("DURATION must be finite and positive")
    if args.STEADY_STATE_STEPS < 0:
        parser.error("STEADY_STATE_STEPS must be nonnegative")
    BOX_SIZE = args.L
    PERSISTENCE_TIME = args.PERSISTENCE_TIME
    STEADY_STATE_STEPS = args.STEADY_STATE_STEPS
    DURATION = args.DURATION
    results = simulate()
    if args.ANIMATE:
        animation = animate(*results)
    separations = results[3]
    if separations.size:
        import matplotlib.pyplot as plt

        plt.figure()
        bins = np.arange(0, separations.max() + 0.05, 0.05)
        plt.hist(separations, bins=bins, edgecolor="black")
        plt.xlabel("Separation / disk diameter")
        plt.ylabel("Count")
        plt.figtext(0.5, 0.02, f"Mean separation = {separations.mean():.3f} disk diameters",
                    ha="center")
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        data_dir = Path(__file__).resolve().parent / "data" / "periodic"
        data_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(data_dir / f"separations_L{BOX_SIZE:g}_tau{PERSISTENCE_TIME:g}.png")
        #plt.show()
