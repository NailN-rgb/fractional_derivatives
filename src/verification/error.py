import numpy as np
from ..core.discretization import Discretization
from ..core.grids import uniform_space, uniform_time
from ..forward.solver import solve
from .reference import ReferenceSolution


def final_error(ref: ReferenceSolution, n_cells: int, n_steps: int,
                final_time: float, stepper: str = "implicit",
                spatially_exact: bool = True) -> float:
    """Максимум |u_h − u| на финальном слое."""
    space = uniform_space(ref.length, n_cells)
    time = uniform_time(final_time, n_steps)
    grid = space if spatially_exact else None

    sol = solve(ref.problem(final_time, grid),
                Discretization(space=space, time=time, stepper=stepper))
    return float(np.abs(sol.final() - ref.exact(space.coords, final_time, grid)).max())


def convergence_study(ref: ReferenceSolution, n_cells: int, steps_list,
                      final_time: float, **kw):
    """Возвращает (taus, errors, orders). Порядки — по соседним парам.

    Пары, а не общий МНК: для решений со слабой особенностью асимптотика
    устанавливается медленно, и единый наклон замаскирует переходный режим.
    """
    errs = np.array([final_error(ref, n_cells, m, final_time, **kw)
                     for m in steps_list])
    taus = final_time / np.asarray(steps_list, dtype=float)
    orders = np.log(errs[:-1] / errs[1:]) / np.log(taus[:-1] / taus[1:])
    return taus, errs, orders


def report(ref, n_cells, steps_list, final_time, **kw) -> str:
    """Таблица для глаз — при отладке полезнее assert'а."""
    taus, errs, orders = convergence_study(ref, n_cells, steps_list, final_time, **kw)
    lines = [f"{'tau':>12} {'error':>12} {'order':>8}"]
    for i, (tau, e) in enumerate(zip(taus, errs)):
        o = f"{orders[i - 1]:8.3f}" if i else " " * 8
        lines.append(f"{tau:12.3e} {e:12.4e} {o}")
    return "\n".join(lines)