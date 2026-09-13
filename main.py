from src.core import Problem, Discretization, uniform_space, uniform_time
from src.forward.solver import solve
import numpy as np

problem = Problem(
    length=1.0, final_time=0.1, alpha=0.9, diffusion=1.0,
    u0=lambda c: np.ones_like(c[0]),
    bc_left=lambda t: 1.0, bc_right=lambda t: 1.0,
)
disc = Discretization(space=uniform_space(1.0, 32),
                      time=uniform_time(0.1, 500), stepper="explicit")

sol = solve(problem, disc)
print(np.abs(sol.u - 1.0).max())     # ожидается ~1e-16