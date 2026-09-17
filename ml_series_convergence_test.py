import numpy as np
from pymittagleffler import mittag_leffler
import matplotlib.pyplot as plt


alpha, beta = 0.5, 1.0
z = np.linspace(-10.0, 1.0, 128)
result = mittag_leffler(z, alpha, beta)

plt.plot(z, result.real, label="Re(E_{0.5,1}(z))")
plt.savefig('plts/ml_series_convergence_test.png')