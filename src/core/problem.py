from dataclasses import dataclass
from ..types import CoeffLike, SpaceFn, SpaceTimeFn, TimeFn

@dataclass(frozen=True, slots=True)
class Problem:
    length: float
    final_time: float
    alpha: float
    diffusion: CoeffLike
    u0: SpaceFn
    bc_left: TimeFn
    bc_right: TimeFn
    source: SpaceTimeFn | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha должна лежать в (0, 1], получено {self.alpha}")
        if self.length <= 0 or self.final_time <= 0:
            raise ValueError("length и final_time должны быть положительны")
        if isinstance(self.diffusion, float) and self.diffusion <= 0:
            raise ValueError("коэффициент диффузии должен быть положителен")

    @property
    def is_classical(self) -> bool:
        return self.alpha == 1.0