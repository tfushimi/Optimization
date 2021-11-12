import numpy as np
from utils import soft_threshold, prox_trace_norm

MAX_COUNT = 5000


def admm(x: np.ndarray,
         gamma: float,
         rho: float):
    L = np.zeros_like(x)
    S = np.zeros_like(x)
    W = np.zeros_like(x)

    x_norm = np.linalg.norm(x)
    for i in range(MAX_COUNT):
        L = prox_trace_norm(x - S - W / rho, rho)
        S = soft_threshold(x - L + W / rho, gamma * rho)
        W = W + rho * (x - L - S)

        res = np.linalg.norm(x - L - S)

        if i % 100 == 0:
            print(f'iteration={i}, diff={res}, norm={x_norm}')

    return L, S


if __name__ == '__main__':
    from PIL import Image

    image = Image.open('../data/picture.jpg').convert('L')
    data = np.asarray(image)
    print(f'row={data.shape[1]}, col={data.shape[1]}')
    low_rank, sparse = admm(data, 0.0001, 1)
    print(data)
    low_rank_image = Image.fromarray(np.uint8(low_rank))
    sparse_image = Image.fromarray(np.uint8(sparse))

    image.save('../data/picture.jpg')
    low_rank_image.save('../output/low_rank_image.jpg')
    sparse_image.save('../output/sparse_image.jpg')
