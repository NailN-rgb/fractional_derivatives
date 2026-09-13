import numpy as np
from ..core.grids import Grid1D, TimeGrid
from ..protocols import SpatialOperator, TimeMemory
from ..types import Array, SpaceTimeFn, TimeFn


class ExplicitStepper:
    """
    r_α = Γ(2-α)·D·τ^α/h² ≤ 1 − 2^{-α}.
    """

    def __init__(
        self,
        memory: TimeMemory,
        operator: SpatialOperator,
        space: Grid1D,
        time: TimeGrid,
        bc_left: TimeFn,
        bc_right: TimeFn,
        source: SpaceTimeFn | None = None,
    ) -> None:
        self.memory = memory
        self.op = operator
        self.space = space
        self.time = time
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.source = source

    def step(self, u_prev: Array, n: int) -> Array:
        t_now = float(self.time.t[n])
        t_prev = float(self.time.t[n - 1])

        kappa = self.memory.diagonal_weight(n)
        rhs = self.memory.history_rhs(n)
        rhs += self.op.matvec(u_prev, t_prev)
        if self.source is not None:
            rhs += np.ravel(self.source(self.space.coords, t_now))

        u = rhs / kappa
        u[0] = self.bc_left(t_now)
        u[-1] = self.bc_right(t_now)
        return u