from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AlphaEstimate:
    alpha: float
    residual: float
    n_solves: int
    bounds: tuple[float, float]


def fit():
    # calculate solution with given alpha 
    # and calculate невязка
    pass

def idetntify_alpha() -> AlphaEstimate:
    # main entry point
    pass

