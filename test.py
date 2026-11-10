import time
import numpy as np
from ordered_opt import approx_ordered

n, delta, eps = 20, 0.5, 0.5          # tes paramètres
w = np.arange(n, 0, -1, dtype=float); w /= w.sum()   # poids linéaires
C = np.random.default_rng(0).integers(1, 100, (n, n)).astype(float)
dummy = lambda M: (np.arange(M.shape[0]), 1.0)       # le nombre de t ne dépend pas du solveur

for alpha in [0.0, 0.5, 1.0]:
    *_, num_t, status = approx_ordered(C, w, delta, eps, min_sum_func=dummy,
                                       alpha=alpha, deadline=time.monotonic() + 30)
    print(alpha, num_t, status)