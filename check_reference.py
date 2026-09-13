import numpy as np
from scipy.special import erfcx
from src.verification.mittag_leffler import (
    mittag_leffler, _integral, _asymptotic, _ASYMPTOTIC_FROM)

print("порог:", _ASYMPTOTIC_FROM)          # должно быть 100.0

x = 9.87
for a in (0.3, 0.5, 0.7):
    print(f"α={a}  диспетчер={mittag_leffler(a, -x):.6f}  "
          f"интеграл={_integral(a, x):.6f}  "
          f"асимпт={_asymptotic(a, -x):.6f}")

print("контроль α=0.5:", erfcx(np.sqrt(x)))   # 0.170434