from typing import Protocol, runtime_checkable
from .types import Array, MemoryCost

@runtime_checkable
class TimeMemory(Protocol):

    memory_cost: MemoryCost

    def reset(self, u0: Array) -> None:
        # Start new calculation
        pass

    def diaginal_weight(self, n: int) -> float:
        # Diaginal coef
        pass

    def history_rhs(self, n:int) -> Array:
        # RHS caclulator
        pass

    def push(self, u: Array, n: int) -> None:
        # Fix caclulated time layer
        pass


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

    