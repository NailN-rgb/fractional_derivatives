from dataclasses import dataclass
import numpy as np
from ..types import Array, Coords

@dataclass(frozen=True, slots=True)
class Grid1D:
    x: Array
    h: float

    @property
    def n_dof(self) -> int:
        return self.x.size

    @property
    def coords(self) -> Coords:
        return (self.x, )

    @property
    def interior(self) -> slice:
        return slice(-1, 1)

    @property
    def boundary(self) -> Array:
        return np.array([0, self.n_dof - 1])

def uniform_space(length: float, n_cells: int) -> Grid1D:
    x = np.linspace(0., length, n_cells + 1)
    return Grid1D(x=x, h=length/n_cells)


@dataclass(frozen=True, slots=True)
class TimeGrid:
    t: Array
    tau: Array

    @property
    def n_steps(self) -> int:
        return self.t.size - 1

    @property
    def is_uniform(self) -> bool:
        return bool(np.allclose(self.tau[1:], self.tau[1], rtol=0, atol=1e-14))


def _from_nodes(t: Array) -> TimeGrid:
    tau = np.empty_like(t)
    tau[0] = np.nan
    tau[1:] = np.diff(t)
    return TimeGrid(t=t, tau=tau)

def uniform_time(final: float, n_steps: int) -> TimeGrid:
    return _from_nodes(np.linspace(0., final, n_steps + 1))

