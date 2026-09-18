import matplotlib.pyplot as plt
import numpy as np

from src.core import Discretization, uniform_space, uniform_time, Problem
from src.forward.solver import solve
from src.verification import FourierReference
from src.viz.render import plot
import src.viz.extract as extract

ALPHAS = [0.1, 0.5, 0.7, 1.0]
FINAL_TIME = 0.1
N_SPASE = 100
N_TIME = 100
L_LENGTH = 1.0
DIFFUSION = 1.0

ax = plt.gca()

for alpha in ALPHAS:
    problem = Problem(
        length=L_LENGTH, final_time=FINAL_TIME, alpha=alpha, diffusion=DIFFUSION,
        u0=lambda c: np.sin(np.pi * c[0]),
        bc_left=lambda t: 0.0,
        bc_right=lambda t: 0.0,
    )

    exact = FourierReference(
        length=L_LENGTH, alpha=alpha, diffusion=DIFFUSION,
        u0_fn=lambda c: np.sin(np.pi * c[0]),
        mode="continuous",
    )

    disc = Discretization(space=uniform_space(L_LENGTH, N_SPASE),
                time=uniform_time(FINAL_TIME, N_TIME), stepper="implicit")

    sol = solve(problem, disc)

    plot(
        extract.error(sol, exact, FINAL_TIME),
    ax, label=f"alpha={alpha}", linestyle='--')

plt.legend()
plt.savefig('plts/different_alphas_1d_test_error.png')
