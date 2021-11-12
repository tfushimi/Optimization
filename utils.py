import numpy as np


def binary_search_step_size(x, z, dx, dz, beta=0.5, prec=0.001):
    """
    Compute the step size in prediction step.
        theta = max{theta | (x + alpha * dx, z + alpha * dz) in N_2(0.5)}
        where N_2(beta) = {(x, z) in F | ||x*z - mu*np.ones(n)||_2 <= beta * mu}, mu = np.dot(x, z)/n}
              F = the set of feasible interior points
        Note that x*z is element-wise multiplication.

    Args:
    x, z: current solutions
    dx, dz: gradient vectors
    beta:
    precision:

    Returns:
    step size in prediction step
    """
    n = len(x)

    # initial low and high of theta
    theta_low = 0.0
    theta_high = 1.0

    # make sure that x + theta * dx > 0 and z + theta * dz > 0
    if len(-x[dx < 0] / dx[dx < 0]) > 0:
        theta_high = np.min([theta_high, np.min(-x[dx < 0] / dx[dx < 0])])
    if len(-z[dz < 0] / dx[dz < 0]) > 0:
        theta_high = np.min([theta_high, np.min(-z[dz < 0] / dz[dz < 0])])

    # compute the updated value with theta = 1.0
    x_high = x + theta_high * dx
    z_high = z + theta_high * dz
    mu_high = np.dot(x_high, z_high) / n

    # return theta = 1.0 if x_high and z_high stay in the N_2(beta)
    if np.linalg.norm(x_high * z_high - mu_high * np.ones(n)) <= beta * mu_high:
        return theta_high

    # run binary search
    while theta_high - theta_low > prec:
        theta_mid = (theta_high + theta_low) / 2
        x_mid = x + theta_mid * dx
        z_mid = z + theta_mid * dz
        mu_mid = np.dot(x_mid, z_mid) / n
        if np.linalg.norm(x_mid * z_mid - mu_mid * np.ones(n)) <= beta * mu_mid:
            theta_low = theta_mid  # discard the lower half
        else:
            theta_high = theta_mid  # discard the upper half
    return theta_low


def soft_threshold(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def prox_nuclear_norm(x: np.ndarray,
                      gamma: float):
    """
    Calculate Soft SVD

    Args:
        x: 2-d array
        gamma: regularization parameter

    Returns:

    """
    u, s, v = np.linalg.svd(x)
    m, n = x.shape
    sigma = np.zeros((m, n))
    for i in range(len(s)):
        sigma[i, i] = np.max(s[i] - gamma, 0)
    return np.dot(np.dot(u, sigma), v)


def prox_group_norm(b: np.ndarray,
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


def prox_trace_norm(x: np.ndarray,
                    gamma: float):
    """
    Calculate Soft SVD

    Args:
        x: 2-d array
        gamma: regularization parameter

    Returns:

    """
    u, s, v = np.linalg.svd(x)
    m, n = x.shape
    sigma = np.zeros((m, n))
    for i in range(len(s)):
        sigma[i, i] = soft_threshold(s[i], gamma)
    return np.dot(np.dot(u, sigma), v)
