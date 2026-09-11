import math 
import numpy as np

from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.special import gamma


alpha = 1

L = 1
N_x = 10
h = L / N_x

T = 1
N_t = 10
tau = T / N_t

def sigma_ta(tau: float):
    return 1 / (math.gamma(2. - alpha) * tau**alpha)

def s_j(j: int):
    J = np.arange(1, j + 1)
    return J ** (1. - alpha) - (J - 1) ** (1 - alpha)

def initial_condition(x: float):
    return np.exp(-(x - L/2)**2)

x_mesh = np.linspace(0, L, N_x + 1)
t_mesh = np.linspace(0, T, N_t + 1)

u = np.zeros((N_x + 1, N_t + 1))
u[0] = initial_condition(x_mesh)

sigma_tau_alpha = sigma_ta(tau)
h2sigma = h**2 * sigma_tau_alpha

main  = np.full(N_x + 1, 2. + h2sigma)
lower = np.full(N_x, -1)
upper = np.full(N_x, -1)

#BC
main[0] = 1.
main[-1] = 1.
upper[0] = 0.
lower[-1] = 0.

A = diags([lower, main, upper], offsets=[-1, 0, 1], format='csc')

lu = splu(A) # factorization

s = s_j(N_t)

for k in range(1, N_t):
    rhs = h2sigma * u[k-1].copy()

    if k >= 2:
        diffs = u[1:k] - u[0:k-1]
        weights = s[1:k][::-1]
        hist = weights @ diffs
        rhs = h2sigma * hist

    rhs[0] = 0.
    rhs[-1] = 0.

    u[k] = lu.solve(rhs)


print("Initialized")
