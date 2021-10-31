import numpy as np
from utils import soft_threshold
from collections import namedtuple

MAX_COUNT = 1000


RegressionResult = namedtuple('RegressionResult', ('beta', 'intercept'))


def proximal_gradient_descent(x: np.ndarray,
                              y: np.ndarray,
                              group: np.ndarray,
                              gamma: float) -> RegressionResult:
    """
    Estimate Lasso model via Proximal Gradient Descent

    Args:
         x: 2-d numpy array of predictors
         y: 1-d numpy array of targets
         group: 1-d numpy array of group ids
         gamma: regularization parameter for L1-norm

    Returns:
        (beta, intercept)
    """
    TOL = 1e-6

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
    beta_prev = beta.copy()

    # pre-compute
    xTy = np.dot(x.T, y)
    xTx = np.dot(x.T, x)

    t = 0.00001  # TODO use backtracking
    for _ in range(MAX_COUNT):
        beta = beta + t * (xTy - np.dot(xTx, beta))
        beta = prox_operator(beta, group, t * gamma)

        beta_norm = np.linalg.norm(beta_prev - beta, 2)

        if beta_norm < TOL:
            break

        beta_prev = beta.copy()

    beta /= x_std
    intercept = y_mean - np.dot(x_mean, beta)

    return RegressionResult(beta, intercept)


def prox_operator(b: np.ndarray,
                  group: np.ndarray,
                  threshold: float) -> np.ndarray:
    result = np.zeros_like(b).astype(np.float64)
    unique_group_ids = np.unique(group)
    for group_id in unique_group_ids:
        target_idx = group == group_id
        target_coef = b[target_idx]
        group_norm = np.linalg.norm(target_coef, 2)
        multiplier = 0 if group_norm == 0 else max(0, 1 - threshold / group_norm)
        result[target_idx] = multiplier * target_coef

    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    crime_data = pd.read_table('../data/crime.txt')
    x = crime_data.iloc[:, 2:7].to_numpy().astype(float)
    y = crime_data.iloc[:, 0].to_numpy().astype(float)
    p = x.shape[1]

    group = np.array([0, 1, 1, 2, 2])
    gammas = np.arange(0, 10000, 100)
    beta_list = np.zeros((len(gammas), p))
    for i, gamma in enumerate(gammas):
        result = proximal_gradient_descent(x, y, group, gamma)
        beta_list[i, :] = result.beta

    plt.xlim(0, 10000)
    plt.ylim(-3, 5)
    for j in range(p):
        plt.plot(gammas, beta_list[:, j], label=f"group={group[j]}")
    plt.hlines(y=0, xmin=0, xmax=10000, linestyles='dashed', colors='black')

    plt.title('Group Lasso via Proximal Gradient Descent (crime.txt)')
    plt.xlabel('Gamma')
    plt.ylabel('Beta')
    plt.legend(loc='lower right')
    plt.savefig('../output/group_lasso_pgd.jpg')
    plt.close()
