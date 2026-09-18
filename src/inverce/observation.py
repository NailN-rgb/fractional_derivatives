import numpy as np

from dataclasses import dataclass

from src.types import Array


@dataclass(frozen=True, slots=True)
class PointHistory:
    """Store the history of a point in time."""
    x_history: Array

    def push_observation(self, x: float) -> None:
        self.x_history = np.append(self.x_history, x)

    def get_last_observation(self) -> float:
        return self.x_history[-1]
