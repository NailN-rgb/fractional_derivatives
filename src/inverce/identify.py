from dataclasses import dataclass

from src.core.discretization import Discretization
from src.core.problem import Problem

@dataclass(frozen=True, slots=True)
class AlphaEstimate:
    alpha: float
    residual: float
    n_solves: int
    bounds: tuple[float, float]


def fit():
    # calculate solution with given alpha 
    # and calculate невязка
    pass

def idetntify_alpha(
    problem: Problem,
    disc: Discretization,
    exact_sol_at_point: float,
    sxact_coord_idx: int,
    alpha_start: float,
    alpha_next: float,
    max_iter: int = 10,
    residual_tol: float = 1e-6
) -> AlphaEstimate:
    # 1. Solve
    # 2. get value at estimated point
    # 3. calculate new alpha
    # 4. check convergence
    # 5. return AlphaEstimate



    return AlphaEstimate(
        alpha=alpha_next
    )

