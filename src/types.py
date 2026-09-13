from typing import Callable, Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]

Coords: TypeAlias = tuple[Array, ...]

# Функции, задающие данные задачи
SpaceFn: TypeAlias = Callable[[Coords], Array]              # u0(x)
SpaceTimeFn: TypeAlias = Callable[[Coords, float], Array]   # f(x, t)
TimeFn: TypeAlias = Callable[[float], float]                # g(t) 

CoeffLike: TypeAlias = float | SpaceFn

MemoryCost: TypeAlias = Literal["quadratic", "linear"]