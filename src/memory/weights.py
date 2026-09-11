import numpy as np
from scipy.special import gamma
from ..types import Array

def l1_weights(alpha: float, n: int) -> Array:
    if not 0. < alpha <= 1:
        raise ValueError("alpha must be in (0, 1)")

    p = 1 - alpha
    k = np.arange(1, n, dtype=np.float64)
    b = np.empty(n)
    b[0] = 1.
    b[1:] = k ** p * np.expm1(p * np.log1p(1. / k))
    return b

def telescoped_coeffs(b: Array, n: int) -> Array:
    c = np.empty(n)
    c[0] = b[n - 1]
    c[1:] = (-np.diff(b[:n]))[::-1]
    return c

def l1_scale(alpha: float, tau: float) -> float:
    return tau ** (-alpha) / gamma(2 - alpha)