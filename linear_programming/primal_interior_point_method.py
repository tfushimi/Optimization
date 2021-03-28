import numpy as np
from copy import deepcopy

class PrimalInteriorPoint(object):
    def __init__(self, c, A, b, init, gamma=0.5, beta=0.5, mu0=1, eps=1e-6):
        # data
        self.c = c
        self.A = A
        self.b = b

        # machine epsilon
        self.eps = eps

        # parameters
        self.gamma = gamma
        self.mu = mu0
        self.beta = beta

        # variables
        self.x = deepcopy(init)
        self.z = np.ones(len(self.x))

        # counter
        self.count = 0

    def _compute_objective(self):
        return np.dot(self.x, self.z)

    def _compute_direction(self):
        raise NotImplementedError

    def solve(self, max_iter=100):
        raise NotImplementedError

class PrimalPathFollowing(PrimalInteriorPoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_direction(self):
        lhs = np.dot(self.A, np.dot(np.diag(self.x ** 2), self.A.T))
        rhs = np.dot(self.A, (self.x ** 2) * (self.c - self.mu / self.x))
        y = np.linalg.solve(lhs, rhs)
        return (1 / self.mu) * np.dot(np.diag(self.x ** 2), np.dot(self.A.T, y) + self.mu / self.x - self.c)

    def solve(self, max_iter=100):
        while True:
            dx = self._compute_direction()
            self.x += dx
            self.z = self.mu * (1 / self.x - dx / self.x ** 2)
            self.mu *= self.gamma
            self.count += 1
            mu = self._compute_objective()
            print("%s iteration: objective value %.5f" % (self.count, mu))
            if mu < self.eps:
                print("Optimal value = %.3f" % np.dot(c, self.x))
                break
            if self.count >= max_iter:
                print("Optimal value is not found")
                break

class PrimalAffineScaling(PrimalInteriorPoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __compute_direction(self):
        lhs = np.dot(self.A, np.dot(np.diag(self.x**2), self.A.T))
        rhs = np.dot(self.A, np.dot(np.diag(self.x**2), self.c))
        y = np.linalg.solve(lhs, rhs)
        z = self.c - np.dot(self.A.T, y)
        return - np.dot(np.diag(self.x**2), z) / np.linalg.norm(np.dot(np.diag(self.x), z))

    def solve(self, max_iter=100):
        while True:
            dx = self.__compute_direction()
            self.x += dx
            self.count += 1
            print("%s iteration: norm of delta x %.5f" % (self.count, np.linalg.norm(dx)))
            if np.linalg.norm(dx) < self.eps:
                print("Optimal value = %.3f" % np.dot(c, self.x))
                break
            if self.count >= max_iter:
                print("Optimal value is not found")
                break

if __name__ == "__main__":
    c = np.array([-1, -1, 0, 0])
    A = np.array([[2, 1, 1, 0], [1, 3, 0, 1]])
    b = np.array([4, 5])
    init = np.hstack([np.ones(4)])

    print("Path-Following")
    path_following = PrimalPathFollowing(c=c, A=A, b=b, init=init)
    path_following.solve()
    x = path_following.x
    print(np.round(x, 3))
    print("Ax = b:", np.all(np.abs(b - np.dot(A, x)) <= 1e-5))

    print("Affine-Scaling")
    affine_scaling = PrimalAffineScaling(c=c, A=A, b=b, init=init)
    affine_scaling.solve()
    x = affine_scaling.x
    print(np.round(x, 3))
    print("Ax = b:", np.all(np.abs(b - np.dot(A, x)) <= 1e-5))