from dataclasses import dataclass

from ..protocols import TimeMemory
from ..core.discretization import Discretization
from ..core.problem import Problem
from ..memory.classical import ClassicalMemory
from ..memory.l1_direct import L1Direct
from ..operators.laplace1d import Laplace1D
from ..protocols import Stepper
from ..solvers.thomas_solver import ThomasSolver
from ..steppers.explicit import ExplicitStepper
from ..steppers.implicit import ImplicitStepper
from .stability import check_explicit

_QUADRATIC_LIMIT = 5e8

@dataclass(frozen=True, slots=True)
class Assembled:
    stepper: Stepper
    memory: TimeMemory

def _build_memory(problem: Problem, disc: Discretization):
    kind = disc.memory
    if kind == "auto":
        kind = "classical" if problem.is_classical else "l1_direct"

    if kind == "classical":
        if not problem.is_classical:
            raise ValueError("ClassicalMemory применима только при alpha = 1")
        return ClassicalMemory(disc.time, disc.space.n_dof)

    if kind == "l1_direct":
        cells = (disc.time.n_steps + 1) * disc.space.n_dof
        if cells > _QUADRATIC_LIMIT:
            raise ValueError(
                f"L1Direct требует {cells * 8 / 2**30:.1f} ГБ истории. "
                f"Уменьшите сетку или дождитесь SOEMemory."
            )
        return L1Direct(disc.time, disc.space.n_dof, problem.alpha)

    raise ValueError(f"неизвестное ядро памяти: {kind}")


def assemble(problem: Problem, disc: Discretization) -> Stepper:
    """Единственное место в проекте, где есть if по типу задачи."""
    operator = Laplace1D(disc.space, problem.diffusion)
    memory = _build_memory(problem, disc)

    common = dict(memory=memory, operator=operator, space=disc.space,
                  time=disc.time, bc_left=problem.bc_left,
                  bc_right=problem.bc_right, source=problem.source)

    if disc.stepper == "explicit":
        if disc.check_stability:
            check_explicit(problem, disc)
        return Assembled(ExplicitStepper(**common), memory=memory)

    if disc.stepper == "implicit":
        return Assembled(ImplicitStepper(solver=ThomasSolver(operator), **common), memory=memory)

    raise ValueError(f"неизвестный степпер: {disc.stepper}")