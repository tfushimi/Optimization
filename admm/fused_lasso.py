import numpy as np
from utils import soft_threshold

MAX_COUNT = 1000


def admm(y: np.ndarray,
         gamma: float,
         rho: float) -> np.ndarray:
    """
    Estimate Fused Lasso model via ADMM

    Args:
         y: 1-d numpy array of targets
         gamma: regularization parameter for L1-norm
         rho: the augmented Lagrangian parameter

    Returns:
        beta
    """
    ABSTOL = 1e-4
    RELTOL = 1e-4

    N = len(y)
    if y.ndim != 1:
        y = y.reshape((N,))

    # initialization
    beta = np.zeros((N,))
    z = np.zeros((N-1,))
    u = np.zeros((N-1,))
    z_prev = z.copy()

    # precompute matrices
    F = np.identity(N) + np.diag(-np.ones(N-1), k=1)
    F = F[:N-1, :]
    A = np.identity(N) + rho * np.dot(F.T, F)

    for _ in range(MAX_COUNT):
        beta = np.linalg.solve(A, y + rho * np.dot(F.T, z - u))
        FBeta = np.dot(F, beta)
        z = soft_threshold(FBeta + u, gamma / rho)
        u = u + FBeta - z

        pri_res = np.linalg.norm(FBeta - z, 2)
        dual_res = np.linalg.norm(-rho * np.dot(F.T, z - z_prev), 2)

        pri_eps = np.sqrt(N) * ABSTOL + RELTOL * np.maximum(np.linalg.norm(FBeta, 2), np.linalg.norm(-z, 2))
        dual_eps = np.sqrt(N) * ABSTOL + RELTOL * np.linalg.norm(rho * np.dot(F.T, u), 2)

        if pri_res < pri_eps and dual_res < dual_eps:
            break

        z_prev = z.copy()

    return beta


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    cgh_data = pd.read_table('../data/cgh.txt')
    y = cgh_data.iloc[:, 0].to_numpy().astype(float)
    N = len(y)

    gammas = np.linspace(1, 10, 3)
    beta_list = np.zeros((len(gammas), N))
    for i, gamma in enumerate(gammas):
        print(f'gamma={gamma}')
        beta_list[i, :] = admm(y, gamma, rho=1.0)

    plt.scatter(range(N), y, s=1, color='black')
    colors = ['red', 'green', 'blue']
    for i, (gamma, color) in enumerate(zip(gammas, colors)):
        plt.plot(range(N), beta_list[i, :], label=f"gamma={gamma}", color=color)

    plt.title('Fused Lasso via ADMM (cgh.txt)')
    plt.xlabel('log2 ratio')
    plt.ylabel('genome order')
    plt.legend(loc='upper right')
    plt.savefig('../output/fused_lasso.jpg')
    plt.close()
