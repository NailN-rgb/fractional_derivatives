from typing import Protocol
from ..core.grids import Grid1D
from ..core.problem import Problem
from ..types import Array, Coords

class ReferenceSolution(Protocol):
    length: float
    alpha: float
    diffusion: float

    def problem(self, final_time: float, grid: Grid1D | None = None) -> Problem: ...

    def exact(self, coords: Coords, t: float, grid: Grid1D | None = None) -> Array: ...

    