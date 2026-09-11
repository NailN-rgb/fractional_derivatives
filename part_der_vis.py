import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma

def caputo_derivative(f, tau, alpha):
    M = len(f) - 1
    D = np.full(M + 1, np.nan)

    k = np.arange(M)
    b = (k + 1)**(1 - alpha) - k**(1 - alpha)
    coef = tau**(-alpha) / gamma(2 - alpha)

    for n in range(1, M + 1):
        diffs = f[1:n+1] - f[0:n]        
        weights = b[0:n][::-1]
        D[n] = coef * np.sum(weights * diffs)

    return D


T = 2 * np.pi
M = 2000
h = T / M
x = np.linspace(0, T, M+1)

f = np.sin(x)

for alpha in np.arange(0.1, 1, 0.2):
    f_partial = caputo_derivative(f, h, alpha)
    plt.plot(x, f_partial, label=rf'$\alpha = {alpha:.1f}$')

plt.legend()
plt.savefig('plts/partial_f.png')