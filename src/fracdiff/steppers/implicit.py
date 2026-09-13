import numpy as np
from ..types import Array

class ImplicitStepper:

    def __init__(self, memory, operator, solver, space, time,
                 bc_left, bc_right, source=None) -> None:
        self.memory = memory
        self.op = operator
        self.space = space
        self.time = time
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.source = source
        self.solver = solver

    def step(self, u_prev: Array, n: int) -> Array:
        t_now = float(self.time.t[n])

        kappa = self.memory.diagonal_weight(n)
        rhs = self.memory.history_rhs(n)

        if self.source is not None:
            rhs += np.ravel(self.source(self.space.coords, t_now))
        rhs /= kappa

        rhs[0] = self.bc_left(t_now)
        rhs[-1] = self.bc_right(t_now)

        return self.solver.solve(1.0 / kappa, rhs, u_ref=u_prev, t=t_now)