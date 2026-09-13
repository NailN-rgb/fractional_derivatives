import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, rgamma

from functools import lru_cache

_ASYMPTOTIC_FROM = 100.0


def mittag_leffler(alpha: float, z: float) -> float:
    if z > 0.0:
        raise ValueError("реализовано только для z <= 0")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha должна быть в (0, 1], получено {alpha}")
    if z == 0.0:
        return 1.0
    if alpha == 1.0:
        return float(np.exp(z))
    if alpha == 0.5:
        return float(erfcx(-z))
    if -z > _ASYMPTOTIC_FROM:
        return _asymptotic(alpha, z)
    return _integral(alpha, -z)


def _integral(alpha: float, x: float) -> float:
    """Интегральное представление. x = −z ≥ 0."""
    c = np.cos(alpha * np.pi)
    pref = np.sin(alpha * np.pi) / (alpha * np.pi)
    inv_a = 1.0 / alpha

    def f(y: float) -> float:
        return np.exp(-x * y**inv_a) / (y * y + 2.0 * y * c + 1.0)

    # масштаб убывания экспоненты ~ x^{-α}; разбиваем там, где она падает
    s0 = min(1.0, x ** (-alpha)) if x > 0 else 1.0
    total = 0.0
    for a, b in ((0.0, s0), (s0, 1.0), (1.0, np.inf)):
        if a < b:
            total += quad(f, a, b, epsabs=1e-14, epsrel=1e-12, limit=200)[0]
    return pref * total


def _asymptotic(alpha: float, z: float, terms: int = 12) -> float:
    k = np.arange(1, terms + 1)
    return float(-np.sum(rgamma(1.0 - alpha * k) / z**k))

@lru_cache(maxsize=100_000)
def _cached(alpha: float, z: float) -> float:
    return mittag_leffler(alpha, z)


def mittag_leffler_vec(alpha: float, z: np.ndarray) -> np.ndarray:
    """Поэлементно с кэшем. В convergence_study повторов тысячи."""
    return np.array([_cached(alpha, float(v)) for v in np.atleast_1d(z)])