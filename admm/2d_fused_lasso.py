import numpy as np
from utils import soft_threshold

MAX_COUNT = 1000


def admm(y, F, gamma, rho=1):
    ABSTOL = 1e-4
    RELTOL = 1e-4

    N = len(y)
    if y.ndim != 1:
        y = y.reshape((N,))

    # initialization
    beta = np.zeros((N,))
    z = np.zeros((F.shape[0],))
    u = np.zeros((F.shape[0],))
    z_prev = z.copy()

    # precompute matrices
    A = np.identity(N) + rho * np.dot(F.T, F)

    for i in range(MAX_COUNT):
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

        print(f'i={i}, pri_res={pri_res}, pri_eps={pri_eps}, dual_res={dual_res}, dual_eps={dual_eps}')

        z_prev = z.copy()

    return beta


def create_tv_mat(nrow: int, ncol: int) -> np.ndarray:
    F = np.zeros((2 * (nrow - 1) * (ncol - 1) + (nrow - 1) + (ncol - 1), nrow * ncol))
    k = 0
    for i in range(nrow):
        for j in range(ncol):
            base = i * ncol + j
            if i < nrow - 1 and j < ncol - 1:
                # horizontal diff
                F[k, base] = 1
                F[k, base + 1] = -1

                # vertical diff
                F[k + 1, base] = 1
                F[k + 1, base + ncol] = -1

                # increment by 2
                k += 2
            elif i < nrow - 1 and j == ncol - 1:
                # vertical diff
                F[k, base] = 1
                F[k, base + ncol] = -1
                k += 1
            elif i == nrow - 1 and j < ncol - 1:
                # horizontal diff
                F[k, base] = 1
                F[k, base + 1] = -1
                k += 1
    return F


if __name__ == '__main__':
    from PIL import Image

    image = Image.open('../data/small_lena.png')

    image_array = np.array(image.convert('L'), 'float')

    nrow, ncol = image_array.shape[0], image_array.shape[1]
    image_array = (image_array + 30 * (np.random.randn(nrow, ncol)))
    image_array[image_array > 255] = 255
    image_array[image_array < 0] = 0

    y = image_array.reshape((-1,))
    F = create_tv_mat(nrow, ncol)
    print(F.shape)
    gamma = 20
    result = admm(y, F, gamma)

    denoised_image_array = result.reshape((nrow, ncol))

    noisy_image = Image.fromarray(np.uint8(image_array))
    noisy_image.save('../output/noisy_image.png')

    denoised_image = Image.fromarray(np.uint8(denoised_image_array))
    denoised_image.save('../output/denoised_image.png')
