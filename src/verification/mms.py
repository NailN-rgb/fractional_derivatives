from dataclasses import dataclass
import numpy as np
from scipy.special import gamma

from ..core.grids import Grid1D
from ..core.problem import Problem
from ..types import Array, Coords


def caputo_power(sigma: float, alpha: float, t: float) -> float:
    return gamma(sigma + 1.0) / gamma(sigma + 1.0 - alpha) * t ** (sigma - alpha)


@dataclass(frozen=True, slots=True)
class SeparableMMS:
    length: float
    alpha: float
    diffusion: float
    sigma: float = 2.0

    def _shape(self, coords: Coords) -> Array:
        return np.sin(np.pi * coords[0] / self.length)

    def _eigenvalue(self, h: float | None) -> float:
        k = np.pi / self.length
        if h is None:
            return -self.diffusion * k**2
        return -self.diffusion * 4.0 / h**2 * np.sin(k * h / 2.0) ** 2

    def exact(self, coords: Coords, t: float, grid: Grid1D | None = None) -> Array:
        return (1.0 + t**self.sigma) * self._shape(coords)

    def _source(self, h: float | None):
        lam = self._eigenvalue(h)

        def f(coords: Coords, t: float) -> Array:
            amp = caputo_power(self.sigma, self.alpha, t) - lam * (1.0 + t**self.sigma)
            return amp * self._shape(coords)

        return f

    def problem(self, final_time: float, grid: Grid1D | None = None) -> Problem:
        return Problem(
            length=self.length, final_time=final_time,
            alpha=self.alpha, diffusion=self.diffusion,
            u0=self._shape,
            bc_left=lambda t: 0.0, bc_right=lambda t: 0.0,
            source=self._source(None if grid is None else grid.h),
        )