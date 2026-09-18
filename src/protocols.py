from typing import Protocol, runtime_checkable

from src.forward.solution import Solution
from .types import Array, MemoryCost, Coords

@runtime_checkable
class TimeMemory(Protocol):

    memory_cost: MemoryCost

    def reset(self, u0: Array) -> None:
        # Start new calculation
        pass

    def diagonal_weight(self, n: int) -> float:
        # Diaginal coef
        pass

    def history_rhs(self, n:int) -> Array:
        # RHS caclulator
        pass

    def push(self, u: Array, n: int) -> None:
        # Fix caclulated time layer
        pass

@runtime_checkable
class Grid(Protocol):
    shape: tuple[int,...]
    n_dof: int
    coords: Coords
    nodes: Coords
    spacing: tuple[float,...]
    axes: tuple[str,...]

@runtime_checkable
class SpatialOperator(Protocol):
    n_dof: int
    is_linear: bool
    is_time_dependent: bool

    def matvec(self, u:Array, t:float) -> Array:
        # A * u
        pass

@runtime_checkable
class ShiftedSolver(Protocol):
    def solve(self, c: float, rhs: Array, u_ref: Array, t: float) -> Array:
        pass

@runtime_checkable
class Stepper(Protocol):
    def step(self, u_prev: Array, n: int) -> Array:
        # return u^k
        pass

@runtime_checkable
class Observation(Protocol):
    def __call__(self, sol: Solution) -> None:
        pass
