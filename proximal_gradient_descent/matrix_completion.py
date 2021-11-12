import numpy as np

from utils import prox_nuclear_norm

MAX_ITER = 1000


def soft_impute(x: np.ndarray,
                mask: np.ndarray,
                *,
                gamma: float,
                rel_tol: float = 1e-4,
                verbose: bool = False):
    """
    Soft-Impute Algorithm

    Args:
        x: 2-d array with missing elements
        mask: 2-d array indicating observed elements
        gamma: regularization parameter
        rel_tol: relative tolerance
        verbose: print progress if True

    Returns:
        soft inputed 2-d array
    """
    sol = x.copy()
    sol_prev = sol.copy()
    mask = mask.astype('uint8')
    for i in range(MAX_ITER):
        sol = prox_nuclear_norm(mask * x + (1 - mask) * sol, gamma)

        abs_eps = np.linalg.norm(sol)
        rel_eps = np.linalg.norm(sol - sol_prev) / abs_eps

        if verbose and (i % 50 == 0):
            print(f'iteration={i}, rel_eps={rel_eps}, abs_eps={abs_eps}, tol={rel_tol}')

        if rel_eps < rel_tol:
            break
        sol_prev = sol.copy()
    return sol


def multi_dim_soft_impute(x: np.ndarray,
                          *,
                          gamma: float,
                          verbose: bool = False):
    """
    Run Soft-Impute Algorithm for 2-d array in each dimension

    Args:
        x: 3-d array
        gamma: regularization
        verbose: print progress if True

    Returns:
        Soft imputed 3-d array
    """
    assert x.ndim == 3
    _, _, d = x.shape
    imputed_data = np.zeros_like(x)
    for i in range(d):
        if verbose:
            print(f'dimension={i}')
        imputed_data[:, :, i] = soft_impute(x[:, :, i], mask, gamma=gamma, verbose=verbose)
    return imputed_data


if __name__ == '__main__':
    from PIL import Image

    image = Image.open('../data/image.jpg')
    data = np.asarray(image)
    nrow, ncol, d = data.shape
    print(f'nrow={nrow}, ncol={ncol}, d={d}')

    mask = np.random.binomial(1, size=nrow * ncol, p=0.8).reshape((nrow, ncol))

    corrupted_data = np.zeros((nrow, ncol, 3))
    for i in range(d):
        corrupted_data[:, :, i] = (mask * data[:, :, i])

    imputed_data = multi_dim_soft_impute(corrupted_data, gamma=10, verbose=True)

    corrupted_image = Image.fromarray(np.uint8(corrupted_data))
    corrupted_image.save('../output/corrupted_image.jpg')

    imputed_image = Image.fromarray(np.uint8(imputed_data))
    imputed_image.save('../output/imputed_image.jpg')
