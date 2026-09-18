import numpy as np
from ..core.grids import TimeGrid
from ..types import Array
from .weights import l1_scale, l1_weights

class L1Direct:
    memory_cost = "quadratic"

    def __init__(self, time: TimeGrid, n_dof: int, alpha: float) -> None:
        if not time.is_uniform:
            raise ValueError("L1Direct требует равномерной сетки по времени")
        self.alpha = alpha
        self.tau = float(time.tau[1])
        self.mu = l1_scale(alpha, self.tau)

        m = time.n_steps
        b = l1_weights(alpha, m)
        self._b_last = b
        self._d = -np.diff(b)
        self._u = np.empty((m + 1, n_dof))
        self._filled = 0

    def reset(self, u0: Array) -> None:
        self._u[0] = u0
        self._filled = 1

    def diagonal_weight(self, n: int) -> float:
        return self.mu

    def history_rhs(self, n: int) -> Array:
        if not 1 <= n <= self._filled:
            raise IndexError(f"история заполнена до слоя {self._filled - 1}, запрошен H_{n}")
        
        out = self._b_last[n - 1] * self._u[0]

        if n > 1:
            tail = self._d[: n - 1][::-1]
            out += tail @ self._u[1:n]
        return self.mu * out

    def push(self, u: Array, n: int) -> None:
        self._u[n] = u
        self._filled = n + 1