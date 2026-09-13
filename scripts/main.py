from src.core import Problem, Discretization, uniform_space, uniform_time
from src.forward.solver import solve
import numpy as np

from src.viz.render import plot
import src.viz.extract as extract

def box(c):
    x = c[0]
    return np.where((0.3 < x) & (x < 0.6), 1.0, 0.0)

problem = Problem(
    length=1.0, final_time=0.1, alpha=0.9, diffusion=1.0,
    u0=box,
    bc_left=lambda t: 0.0, bc_right=lambda t: 0.0,
)
disc = Discretization(space=uniform_space(1.0, 32),
                      time=uniform_time(0.1, 500), stepper="explicit")

sol = solve(problem, disc)
print(np.abs(sol.u - 1.0).max())

plot(extract.at(sol, 0),).figure.savefig('../plts/viz.png')