import numpy as np
from utils import soft_threshold
from collections import namedtuple

MAX_COUNT = 1000


RegressionResult = namedtuple('RegressionResult', ('beta', 'intercept'))


def admm(x: np.ndarray,
         y: np.ndarray,
         gamma: float,
         rho: float) -> RegressionResult:
    """
    Estimate Lasso model via ADMM

    Args:
         x: 2-d numpy array of predictors
         y: 1-d numpy array of targets
         gamma: regularization parameter for L1-norm
         rho: the augmented Lagrangian parameter

    Returns:
        (beta, intercept)
    """
    ABSTOL = 1e-4
    RELTOL = 1e-4

    n_row, n_col = x.shape
    if y.ndim != 1:
        y = y.reshape((n_row,))

    # normalize x and centralize y
    x_std = np.std(x, axis=0)
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y)
    x = (x - x_mean) / x_std
    y = y - y_mean

    # initialization
    beta = np.zeros((n_col,))
    z = np.zeros((n_col,))
    u = np.zeros((n_col,))
    z_prev = z.copy()

    # precompute matrices
    A = np.dot(x.T, x) + rho * np.identity(n_col)
    xTy = np.dot(x.T, y)

    for _ in range(MAX_COUNT):
        beta = np.linalg.solve(A, xTy + rho * (z - u))
        z = soft_threshold(beta + u, gamma / rho)
        u = u + beta - z

        pri_res = np.linalg.norm(beta - z, 2)
        dual_res = np.linalg.norm(-rho * (z - z_prev), 2)

        pri_eps = np.sqrt(n_col) * ABSTOL + RELTOL * np.maximum(np.linalg.norm(beta, 2), np.linalg.norm(-z, 2))
        dual_eps = np.sqrt(n_col) * ABSTOL + RELTOL * np.linalg.norm(rho * u, 2)

        if pri_res < pri_eps and dual_res < dual_eps:
            break

        z_prev = z.copy()

    beta /= x_std
    intercept = y_mean - np.dot(x_mean, beta)

    return RegressionResult(beta, intercept)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    crime_data = pd.read_table('../data/crime.txt')
    x = crime_data.iloc[:, 2:7].to_numpy().astype(float)
    y = crime_data.iloc[:, 0].to_numpy().astype(float)
    p = x.shape[1]

    gammas = np.arange(0, 10000, 100)
    beta_list = np.zeros((len(gammas), p))
    for i, gamma in enumerate(gammas):
        result = admm(x, y, gamma, rho=1.0)
        beta_list[i, :] = result.beta

    plt.xlim(0, 10000)
    plt.ylim(-10, 15)
    for j in range(p):
        plt.plot(gammas, beta_list[:, j])

    plt.title('LASSO regression via ADMM (crime.txt)')
    plt.xlabel('Gamma')
    plt.ylabel('Beta')
    plt.savefig('../output/linear_lasso_admm.jpg')
    plt.close()
