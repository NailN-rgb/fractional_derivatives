from dataclasses import dataclass
from typing import Literal
from .grids import Grid1D, TimeGrid

StepperKind = Literal["explicit", "implicit"]
MemoryKind = Literal["l1_direct", "classical", "auto"]


@dataclass(frozen=True, slots=True)
class Discretization:
    space: Grid1D
    time: TimeGrid
    stepper: StepperKind = "implicit"
    memory: MemoryKind = "auto"
    tol: float = 1e-10
    check_stability: bool = True