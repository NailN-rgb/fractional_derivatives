from dataclasses import dataclass
import numpy as np
from scipy.fft import dst, idst

from ..core.grids import Grid1D
from ..core.problem import Problem
from ..types import Array, Coords, SpaceFn
from .mittag_leffler import mittag_leffler_vec


@dataclass(frozen=True, slots=True)
class FourierReference:
    """
    mode='discrete'  
    mode='continuous' 
    """

    length: float
    alpha: float
    diffusion: float
    u0_fn: SpaceFn
    mode: str = "discrete"
    n_modes: int = 200

    def _eigenvalues(self, n: np.ndarray, h: float) -> np.ndarray:
        k = n * np.pi / self.length
        if self.mode == "continuous":
            return -self.diffusion * k**2
        return -self.diffusion * 4.0 / h**2 * np.sin(k * h / 2.0) ** 2

    def exact(self, coords: Coords, t: float, grid: Grid1D | None = None) -> Array:
        if grid is None:
            raise ValueError("FourierReference требует grid")

        g = self.u0_fn(grid.coords)[1:-1]
        c = dst(g, type=1, norm="ortho")               # коэффициенты c_n
        n = np.arange(1, g.size + 1)
        lam = self._eigenvalues(n, grid.h)

        decay = 1.0 if t == 0.0 else mittag_leffler_vec(self.alpha, lam * t**self.alpha)
        out = np.zeros(grid.n_dof)
        out[1:-1] = idst(c * decay, type=1, norm="ortho")
        return out

    def problem(self, final_time: float, grid: Grid1D | None = None) -> Problem:
        return Problem(
            length=self.length, final_time=final_time,
            alpha=self.alpha, diffusion=self.diffusion,
            u0=self.u0_fn,
            bc_left=lambda t: 0.0, bc_right=lambda t: 0.0,
            source=None,
        )