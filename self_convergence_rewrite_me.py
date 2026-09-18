import numpy as np

from src.core import Discretization, Problem, uniform_space, uniform_time
from src.forward.solver import solve

FINAL_TIME = 1.0

def final_layer(n_cells: int, n_steps: int, alpha: float) -> np.ndarray:
    problem = Problem(
        length=1.0, final_time=FINAL_TIME, alpha=alpha, diffusion=1.0,
        u0=lambda c: np.sin(np.pi * c[0]),
        bc_left=lambda t: 0.0, bc_right=lambda t: 0.0,
    )
    disc = Discretization(space=uniform_space(1.0, n_cells),
                          time=uniform_time(FINAL_TIME, n_steps))
    return solve(problem, disc).final()


def report(title: str, levels: list[int], diffs: list[float]) -> str:
    d = np.asarray(diffs)
    orders = np.log2(d[:-1] / d[1:])
    lines = [title, f"{'сетки':>16} {'d':>12} {'p':>8}"]
    for i, (a, b) in enumerate(zip(levels[:-1], levels[1:])):
        p = f"{orders[i - 1]:8.3f}" if i else " " * 8
        lines.append(f"{a:8d} -> {b:4d} {d[i]:12.4e} {p}")
    return "\n".join(lines)


def space_order(alpha: float, cells: list[int], n_steps: int = 200) -> str:
    u = {n: final_layer(n, n_steps, alpha) for n in cells}

    diffs = [np.abs(u[a] - u[b][::2]).max() for a, b in zip(cells[:-1], cells[1:])]
    return report(f"alpha={alpha}: по h (n_steps={n_steps})", cells, diffs)


def time_order(alpha: float, steps: list[int], n_cells: int = 32) -> str:
    """h фиксировано; финальный слой у всех сеток лежит в одном t = T."""
    u = {n: final_layer(n_cells, n, alpha) for n in steps}
    diffs = [np.abs(u[a] - u[b]).max() for a, b in zip(steps[:-1], steps[1:])]
    return report(f"alpha={alpha}: по tau (n_cells={n_cells})", steps, diffs)


if __name__ == "__main__":
    CELLS = [10, 20, 40, 80, 160, 320, 640]
    STEPS = [25, 50, 100, 200, 400, 800, 1600, 3200]

    for alpha in (1.0, 0.3):
        print(space_order(alpha, CELLS), "\n")
        print(time_order(alpha, STEPS), "\n")



