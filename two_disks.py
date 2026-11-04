"""Two active disks in a circular arena, inspired by cell_simulation.

Run: python3 two_disks.py (requires numpy and matplotlib).
Lengths below are in disk diameters: D=10 physically, arena diameter=6D=60.
Time and motion parameters are illustrative, not fitted to experiments.

Both disks propel themselves and diffuse. On contact, both turn away;
the green disk temporarily reduces its angular diffusion (more persistence).
Disk repulsion is a soft spring. A hard circular wall blocks outward motion
without changing the heading. There are no periodic boundaries or adhesion.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# Edit these parameters to explore the model. Mobility is set to one.
DISK_DIAMETER = 10.0                 # Physical length represented by one unit
ARENA_DIAMETER = 6.0                # In disk diameters
SPEED = 1.0                         # Disk diameters per time unit
REPULSION = 100.0                   # Spring strength / mobility
TRANSLATIONAL_DIFFUSION = 0.02      # Disk diameters squared per time unit
ANGULAR_DIFFUSION = 0.5             # Radians squared per time unit
PERSISTENT_DIFFUSION = 0.01         # Green disk after contact
PERSISTENCE_TIME = 2.0
DT = 0.001
DURATION = 20.0
SAVE_EVERY = 40                     # One animation frame per 40 steps
SEED = 7


def simulate():
    """Return times, positions (frame, disk, xy), and green persistence flags."""
    rng = np.random.default_rng(SEED)
    # Disk 0 is green, disk 1 is red; initially they face each other.
    positions = np.array([[-1.0, 0.0], [1.0, 0.0]])
    angles = np.array([0.0, np.pi])
    persistent_left = 0.0
    # Disk centers must stay half a disk diameter inside the arena wall.
    center_limit = ARENA_DIAMETER / 2 - 0.5
    times, tracks, persistent = [0.0], [positions.copy()], [False]

    for step in range(1, round(DURATION / DT) + 1):
        persistent_left = max(0.0, persistent_left - DT)
        separation = positions[0] - positions[1]
        distance = np.linalg.norm(separation)
        # A fallback direction avoids division by zero at coincident centers.
        normal = separation / distance if distance > 1e-12 else np.array([1., 0.])
        forces = np.zeros((2, 2))

        if distance < 1.0:  # Contact: one disk diameter between centers
            forces[0] = REPULSION * (1.0 - distance) * normal
            forces[1] = -forces[0]
            away = np.arctan2(normal[1], normal[0])
            angles[:] = [away, away + np.pi]
            # Like the repo, contacts do not refresh an active persistence timer.
            if persistent_left == 0.0:
                persistent_left = PERSISTENCE_TIME

        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        # Euler-Maruyama: drift * dt + sqrt(2 * diffusion * dt) * normal noise.
        positions += (SPEED * directions + forces) * DT
        positions += np.sqrt(2 * TRANSLATIONAL_DIFFUSION * DT) * rng.normal(size=(2, 2))

        # Project centers back onto the allowed circle: a sliding hard wall.
        radii = np.linalg.norm(positions, axis=1)
        outside = radii > center_limit
        positions[outside] *= (center_limit / radii[outside])[:, None]

        angular_diffusion = np.array([
            PERSISTENT_DIFFUSION if persistent_left > 0 else ANGULAR_DIFFUSION,
            ANGULAR_DIFFUSION,
        ])
        angles += np.sqrt(2 * angular_diffusion * DT) * rng.normal(size=2)

        if step % SAVE_EVERY == 0 or step == round(DURATION / DT):
            times.append(step * DT)
            tracks.append(positions.copy())
            persistent.append(persistent_left > 0)

    return np.array(times), np.array(tracks), np.array(persistent)


def animate(times, tracks, persistent):
    """Show the arena; purple means the green disk has increased persistence."""
    fig, ax = plt.subplots(figsize=(6, 6))
    radius = ARENA_DIAMETER / 2
    ax.add_patch(Circle((0, 0), radius, fill=False, color="black"))
    ax.set(xlim=(-radius - .2, radius + .2), ylim=(-radius - .2, radius + .2),
           xlabel="x / disk diameter", ylabel="y / disk diameter", aspect="equal")
    disks = [Circle(tracks[0, i], .5, color=color, alpha=.8)
             for i, color in enumerate(["tab:green", "tab:red"])]
    for disk in disks:
        ax.add_patch(disk)
    title = ax.set_title("")

    def update(frame):
        for i, disk in enumerate(disks):
            disk.center = tracks[frame, i]
        disks[0].set_color("tab:purple" if persistent[frame] else "tab:green")
        title.set_text(f"t = {times[frame]:.2f} | arena diameter = 6D, D = {DISK_DIAMETER:g}")
        return *disks, title

    animation = FuncAnimation(fig, update, frames=len(times), interval=40, blit=True)
    plt.show()
    return animation


if __name__ == "__main__":
    animation = animate(*simulate())
