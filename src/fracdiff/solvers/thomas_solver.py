import numpy as np
from ..linalg.thomas import thomas_lapack
from ..operators.laplace1d import Laplace1D
from ..types import Array


class ThomasSolver:
    def __init__(self, operator: Laplace1D) -> None:
        self.op = operator
        n = operator.n_dof
        self._diag_a = operator.diagonal()
        self._off_a = operator.offdiagonal()
        self._lower = np.empty(n)
        self._upper = np.empty(n)
        self._diag = np.empty(n)

    def solve(self, c: float, rhs: Array, u_ref: Array, t: float) -> Array:
        np.multiply(self._diag_a, -c, out=self._diag)
        self._diag += 1.0

        self._lower.fill(-c * self._off_a)
        self._upper.fill(-c * self._off_a)
        self._lower[0] = 0.0
        self._upper[-1] = 0.0

        self._upper[0] = 0.0
        self._lower[-1] = 0.0

        return thomas_lapack(self._lower, self._diag, self._upper, rhs)