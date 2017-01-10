

from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import numpy as np


def knr(X,y):
    param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                  "kernel": [ExpSineSquared(l, p)
                             for l in np.logspace(-2, 2, 10)
                             for p in np.logspace(0, 2, 10)]}
    kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
    kr.fit(X,y)
    return kr