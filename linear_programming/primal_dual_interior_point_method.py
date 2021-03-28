import numpy as np
from copy import deepcopy
from linear_programming.utils import binary_search_step_size

class PrimalDualInteriorPoint(object):
    """
    Primal-Dual Interior Point method solve both primal and dual linear problems such that
        Primal: minimize c^Tx
                subject to Ax = b, x >= 0
        Dual: maximize b^Ty
              subject to A^Ty + z = c, z >= 0
    """
    def __init__(self, c, A, b, init, eps=1.0e-6):
        # data
        self.c = c
        self.A = A
        self.b = b

        # self.m = the number of equality constraints
        # self.n = the dimension of x
        self.m, self.n = A.shape

        # machine epsilon
        self.eps = eps

        # counting iteration
        self.count = 0

        # variables
        self.x = deepcopy(init[:self.n]) # primal variables
        self.y = deepcopy(init[self.n:self.n+self.m]) # dual variables
        self.z = deepcopy(init[-self.n:]) # slack variables

    def _compute_step_size(self, dx, dz):
        """
        Find a step_size such that x <- x + step_size * dx and z <- z + step_size * dz.
        It is determined so that the new x and z remain in the interior of the feasible region.

        Args:
        dx, dz: the directions in which x and z are updated

        Returns:
        a step size
        """
        raise NotImplementedError

    def _compute_objective(self):
        """
        Compute current objective value

        Returns:
        current objective value
        """
        return np.dot(self.x, self.z) / len(self.x)

    def _compute_directions(self, *args, **kwargs):
        """
        Compute the direction in which variables are updated.

        Returns:
        dx, dy, dz
        """
        raise NotImplementedError

    def solve(self, max_iter=100):
        """
        Run an interior-point method.
        The algorithm terminates until x^Tz < eps or count > max_iter

        Args:
        max_iter: maximum iteration

        Returns
        optimal values of x, y, and z
        """
        raise NotImplementedError

class PrimalDualAffineScaling(PrimalDualInteriorPoint):
    """
    Implements Primal Dual Affine Scaling
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu0 = np.max(self.x * self.z)
        self.K = self.n * np.ceil(np.log(self.n * mu0 / self.eps))

    def _compute_step_size(self, dx, dz, prec=1.0e-6):
        beta = (self.count / self.K)
        return binary_search_step_size(self.x, self.z, dx, dz, beta, prec)

    def _compute_directions(self):
        dy = np.linalg.solve(np.dot(self.A, np.dot(np.diag(self.x / self.z), self.A.T)), self.b)
        dz = -np.dot(self.A.T, dy)
        dx = -self.x / self.z * dz - self.x
        return dx, dy, dz

    def solve(self, max_iter=100):
        while True:
            # compute the directions
            dx, dy, dz = self._compute_directions()

            # find a step_size
            step_size = self._compute_step_size(dx, dz)

            # update variables
            self.x += step_size * dx
            self.y += step_size * dy
            self.z += step_size * dz

            # compute the objective value
            mu = self._compute_objective()

            # increment counter
            self.count += 1

            # print
            print("%sth iteration: step_size = %.5f, objective value = %.5f" % (self.count, step_size, mu))

            # conditions to terminate
            if mu < self.eps:
                print("Optimal value = %.3f" % np.dot(c, self.x))
                print("Optimal value (dual) = %.3f" % np.dot(b, self.y))
                break

            if self.count > max_iter:
                print("Optimal value is not found")
                break


class PrimalDualPathFollowing(PrimalDualInteriorPoint):
    """
    Implement Primal Dual Path Following with Predictor-Corrector method
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_step_size(self, dx, dz, prec=1.0e-6):
        beta = 0.5
        return binary_search_step_size(self.x, self.z, dx, dz, beta, prec)

    def _compute_directions(self, gamma=0):
        rhs = np.dot(self.A, np.dot(np.diag(self.x / self.z), self.A.T))
        mu = self._compute_objective()
        lhs = self.b - gamma * mu * np.dot(self.A, 1 / self.z)
        dy = np.linalg.solve(rhs, lhs)
        dz = -np.dot(self.A.T, dy)
        dx = -self.x / self.z * dz + gamma * mu * (1 / self.z) - self.x
        return dx, dy, dz

    def solve(self, max_iter=100):
        while True:
            # predictor step
            dx, dy, dz = self._compute_directions()

            # find a step_size
            step_size = self._compute_step_size(dx, dz)

            # update variables
            self.x += step_size * dx
            self.y += step_size * dy
            self.z += step_size * dz

            # corrector step
            dx, dy, dz = self._compute_directions(gamma=1)

            # update variables
            self.x += dx
            self.y += dy
            self.z += dz

            # compute the objective value
            mu = self._compute_objective()

            # increment counter by 1
            self.count += 1

            # print
            print("%sth iteration: step_size = %.5f, objective value = %.5f" % (self.count, step_size, mu))

            # conditions to terminate
            if mu < self.eps:
                print("Optimal value = %.3f" % np.dot(c, self.x))
                print("Optimal value (dual) = %.3f" % np.dot(b, self.y))
                break

            if self.count > max_iter:
                print("Optimal value is not found")
                break


if __name__ == "__main__":
    c = np.array([-1, -1, 0, 0])
    A = np.array([[2, 1, 1, 0], [1, 3, 0, 1]])
    b = np.array([4, 5])
    init = np.hstack([np.ones(4), np.array([-1,-1]), np.array([2, 3, 1, 1])])

    print("Affine Scaling")
    affine_scaling = PrimalDualAffineScaling(c=c, A=A, b=b, init=init)
    affine_scaling.solve()
    x, y, z = affine_scaling.x, affine_scaling.y, affine_scaling.z
    print(np.round(x, 3))
    print(np.round(y, 3))
    print(np.round(z, 3))
    print("Ax = b:", np.all(np.abs(b - np.dot(A, x)) <= 1e-5))
    print("A^Ty + z = c:", np.all(np.abs(c - np.dot(A.T, y) - z) <= 1e-5))


    print("Path-Following")
    path_following = PrimalDualPathFollowing(c=c, A=A, b=b, init=init)
    path_following.solve()
    x, y, z = path_following.x, path_following.y, path_following.z
    print(np.round(x, 3))
    print(np.round(y, 3))
    print(np.round(z, 3))
    print("Ax = b:", np.all(np.abs(b - np.dot(A, x)) <= 1e-5))
    print("A^Ty + z = c:", np.all(np.abs(c - np.dot(A.T, y) - z) <= 1e-5))
