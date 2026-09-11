import numpy as np
from ..core.grids import TimeGrid
from ..types import Array


class ClassicalMemory:
    memory_cost = "linear"

    def __init__(self, time: TimeGrid, n_dof: int) -> None:
        if not time.is_uniform:
            raise ValueError("ClassicalMemory требует равномерной сетки по времени")
        self.tau = float(time.tau[1])
        self._prev = np.empty(n_dof)

    def reset(self, u0: Array) -> None:
        self._prev[:] = u0

    def diagonal_weight(self, n: int) -> float:
        return 1.0 / self.tau

    def history_rhs(self, n: int) -> Array:
        return self._prev / self.tau

    def push(self, u: Array, n: int) -> None:
        self._prev[:] = u