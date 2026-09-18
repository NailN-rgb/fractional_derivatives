import matplotlib.pyplot as plt
import numpy as np

from src.core import Discretization, uniform_space, uniform_time
from src.forward.solver import solve
from src.verification import FourierReference
from src.viz.render import plot
import src.viz.extract as extract

ALPHA, FINAL_TIME = 0.7, 0.1

# Эталон — единственный источник и задачи, и аналитики: они не могут разъехаться.
ref = FourierReference(
    length=1.0, alpha=ALPHA, diffusion=1.0,
    u0_fn=lambda c: np.sin(np.pi * c[0]),
    mode="continuous",
)

disc = Discretization(space=uniform_space(ref.length, 100),
                      time=uniform_time(FINAL_TIME, 50), stepper="implicit")

sol = solve(ref.problem(FINAL_TIME), disc)

err = extract.error(sol, ref, FINAL_TIME)
print(f"max|u_h - u| = {np.abs(err.values).max():.3e}")

ax = plt.gca()
plot(extract.at(sol, FINAL_TIME), ax)
plot(extract.exact(ref, disc.space, FINAL_TIME), ax, linestyle='--')
ax.set_title(f"alpha={ALPHA}")
ax.legend()
ax.figure.savefig('plts/viz.png')
