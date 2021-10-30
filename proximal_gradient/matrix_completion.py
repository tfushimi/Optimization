import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow


def soft_svd(lambd, z):
    u, s, vh = np.linalg.svd(z)
    sigma = np.zeros((z.shape[0], z.shape[1]))
    for i in range(len(s)):
        sigma[i, i] = np.max(s[i] - lambd, 0)
    return np.dot(np.dot(u, sigma), vh)


def mat_lasso(lambd, z, mask):
    m = z.shape[0]
    n = z.shape[1]
    guess = np.random.normal(size=m*n).reshape(m, -1)
    for i in range(20):
        guess = soft_svd(lambd, mask * z + (1 - mask) * guess)
    return guess


if __name__ == '__main__':
    image = np.array(Image.open("../output/lion.jpg"))
    m = image[:, :, 1].shape[0]
    n = image[:, :, 1].shape[1]
    p = 0.5
    lambd = 1.0
    mat = np.zeros((m, n, 3))
    mask = np.random.binomial(1, p, size=m*n).reshape(-1, n)
    for i in range(3):
        mat[:, :, i] = mat_lasso(lambd, image[:, :, i], mask)
    Image.fromarray(np.uint8(mat)).save("../output/lion3_compressed_mat_soft.jpg")
    i = Image.open("../output/lion3_compressed_mat_soft.jpg")
    imshow(i)
