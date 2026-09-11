import numpy as np
from ..core.grids import Grid1D, TimeGrid
from ..types import Array


class Solution:
    def __init__(self, u: Array, space: Grid1D, time: TimeGrid) -> None:
        self.u = u
        self.space = space
        self.time = time

    def at_time(self, t: float) -> Array:
        n = int(np.searchsorted(self.time.t, t))
        if n == 0:
            return self.u[0].copy()
        t0, t1 = self.time.t[n - 1], self.time.t[n]
        w = (t - t0) / (t1 - t0)
        return (1.0 - w) * self.u[n - 1] + w * self.u[n]

    def interpolate(self, x: Array, t: float) -> Array:
        return np.interp(x, self.space.x, self.at_time(t))

    def final(self) -> Array:
        return self.u[-1]