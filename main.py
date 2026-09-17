import matplotlib.pyplot as plt

from src.core import Problem, Discretization, uniform_space, uniform_time
from src.forward.solver import solve
from src.viz.render import plot
import src.viz.extract as extract

import numpy as np
from pymittagleffler import mittag_leffler

from src.viz.primitives import Field1D


def box(c):
    x = c[0]
    return np.sin(np.pi * x) / 1

def exact_solution(x: np.ndarray, t: float, problem: Problem):
    x = x[0]
    return mittag_leffler(- problem.diffusion * (np.pi/ problem.length)**2 
                          * t**problem.alpha, problem.alpha, 1.) * \
        np.sin(np.pi * x / problem.length)


problem = Problem(
    length=1.0, final_time=0.1, alpha=1, diffusion=1.0,
    u0=box,
    bc_left=lambda t: 0.0, bc_right=lambda t: 0.0,
)

disc = Discretization(space=uniform_space(1.0, 100),
                      time=uniform_time(0.1, 50), stepper="implicit")

sol = solve(problem, disc)

exact_sol = exact_solution(disc.space.coords, problem.final_time, problem) 
ex_sol = Field1D(
    x=disc.space.coords[0],
    values=exact_sol,
    xlabel='x',
    vlabel='u',
    label='Exact solution'
)

print(np.abs(sol.u[-1] - exact_sol).max())

ax = plt.gca()
plot(extract.at(sol, problem.final_time), ax)
plot(ex_sol, ax, linestyle='--')
ax.set_title(f"alpha={problem.alpha}")
ax.legend()
ax.figure.savefig('plts/viz.png')