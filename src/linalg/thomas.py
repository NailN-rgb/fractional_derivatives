import numpy as np
from ..types import Array
from scipy.linalg import solve_banded


def thomas(lower: Array, diag: Array, upper: Array, rhs: Array) -> Array:
    n = diag.size
    c = np.empty(n)
    d = np.empty(n)

    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom

    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def thomas_lapack(lower: Array, diag: Array, upper: Array, rhs: Array) -> Array:
    """Faster than naive realization"""
    n = diag.size
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)