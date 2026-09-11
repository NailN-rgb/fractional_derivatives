import numpy as np
from ..core.discretization import Discretization
from ..core.problem import Problem
from .assembler import assemble
from .solution import Solution


class ForwardSolver:
    def __init__(self, problem: Problem, disc: Discretization) -> None:
        self.problem = problem
        self.disc = disc
        built = assemble(problem, disc)
        self.stepper = built.stepper
        self.memory = built.memory

    def solve(self) -> Solution:
        space, time = self.disc.space, self.disc.time
        u = np.empty((time.t.size, space.n_dof))

        u[0] = self.problem.u0(space.coords)
        u[0, 0] = self.problem.bc_left(float(time.t[0]))
        u[0, -1] = self.problem.bc_right(float(time.t[0]))

        self.memory.reset(u[0])

        for n in range(1, time.t.size):
            u[n] = self.stepper.step(u[n - 1], n)
            self.memory.push(u[n], n)

        return Solution(u=u, space=space, time=time)


def solve(problem: Problem, disc: Discretization) -> Solution:
    return ForwardSolver(problem, disc).solve()