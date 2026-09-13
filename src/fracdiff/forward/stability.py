import numpy as np

from scipy.special import gamma
from ..core.discretization import Discretization
from ..core.problem import Problem


class StabilityError(RuntimeError):
    """Явная схема нарушает условие устойчивости."""


def fractional_courant(problem: Problem, disc: Discretization) -> float:
    """r_α = Γ(2-α)·D·τ^α / h²."""
    tau = float(disc.time.tau[1])
    h = disc.space.h
    return gamma(2.0 - problem.alpha) * problem.diffusion * tau**problem.alpha / h**2


def stability_threshold(alpha: float) -> float:
    """r_α ≤ 1 − 2^{-α}. При α=1 даёт классические 0.5."""
    return 1.0 - 2.0 ** (-alpha)


def check_explicit(problem: Problem, disc: Discretization) -> None:
    r = fractional_courant(problem, disc)
    limit = stability_threshold(problem.alpha)
    if r > limit:
        tau_max = (limit * disc.space.h**2
                   / (gamma(2.0 - problem.alpha) * problem.diffusion)) ** (1.0 / problem.alpha)
        raise StabilityError(
            f"r_α = {r:.4f} > {limit:.4f} при α = {problem.alpha}. "
            f"Максимальный шаг: τ ≤ {tau_max:.3e} "
            f"(сейчас {float(disc.time.tau[1]):.3e}, нужно не менее "
            f"{int(np.ceil(problem.final_time / tau_max))} шагов). "
            f"Либо используйте stepper='implicit'."
        )