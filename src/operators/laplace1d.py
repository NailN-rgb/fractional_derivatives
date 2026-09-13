import numpy as np
from ..core.grids import Grid1D
from ..types import Array


class Laplace1D:
    """
    Граничные строки — нулевые: значения Дирихле лежат в u
    на позициях 0 и n_dof-1 
    """

    is_linear = True
    is_time_dependent = False

    def __init__(self, grid: Grid1D, diffusion: float) -> None:
        if diffusion <= 0:
            raise ValueError("коэффициент диффузии должен быть положителен")
        self.grid = grid
        self.d = diffusion
        self.n_dof = grid.n_dof
        self._scale = diffusion / grid.h**2

    def matvec(self, u: Array, t: float) -> Array:
        out = np.zeros_like(u)
        out[1:-1] = self._scale * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        return out

    # ── данные для ShiftedSolver ──────────────────────────────
    def diagonal(self) -> Array:
        d = np.full(self.n_dof, -2.0 * self._scale)
        d[0] = d[-1] = 0.0
        return d

    def offdiagonal(self) -> float:
        """Внедиагональный элемент (одинаков для обеих поддиагоналей)."""
        return self._scale