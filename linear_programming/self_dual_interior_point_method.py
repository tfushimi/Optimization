import numpy as np
from linear_programming.utils import binary_search_step_size

class SelfDualPrimalDualPathFollowing:
    def __init__(self, c, A, b, eps=1e-6):
        # A is in shape of (self.m, self.n)
        self.m, self.n = A.shape

        # convert to self-dual LP
        (M0, q0) = self._make_Mq_from_cAb(c, A, b)

        # create an artificial problem and initial values
        (self.M, self.q, self.x, self.z) = self._make_artProb_initialPoint(M0, q0)
        self.primal_vars = None
        self.dual_vars = None

        # machine epsilon
        self.eps = eps

        # counting iteration
        self.count = 0

    def _make_Mq_from_cAb(self, c, A, b):
        """
        Convert a LP problem into a Self-Dual LP problem such that
            minimize q^Tx
            subject to Mx + q = z, x >= 0, z>= 0
            where M = -M.T, q >= 0

        M, x, and q are defined as follows.
        M = [[0, -A, b],
             [A.T, 0, -c],
             [-b.T, c.T, 0]]
        x = [y, x, tau]
        q = 0

        Args:
        c: np.array of shape (n,)
        A: np.array of shape (m, n)
        b: np.array of shape (m,)

        Outputs:
        M: np.array of shape (m+n+1, m+n+1)
        q: np.array of shape (m+n+1,)
        """
        m1 = np.hstack((np.zeros((self.m, self.m)), -A, b.reshape(self.m, -1)))
        m2 = np.hstack((A.T, np.zeros((self.n, self.n)), -c.reshape(self.n, -1)))
        m3 = np.append(np.append(-b, c), 0)
        M = np.vstack((m1, m2, m3))
        q = np.zeros(self.m + self.n + 1)
        return M, q

    def _make_artProb_initialPoint(self, M0, q0):
        """
        Convert a Self-Dual LP problem into an equivalent Self-Dual LP problem which has an interior point.
        The interior point can be used as an initial value

        Args:
        M0: np.array of shape (k, k)
        q0: np.array of shape (k,)

        Returns:
        MM: np.array of shape (N, N)
        qq: np.array of shape (N,)
        x_init: np.array of shape (N,)
        z_init: np.array of shape (N,)
        """
        k = M0.shape[0]
        x0 = np.ones(k)
        mu0 = np.dot(q0, x0) / (k + 1) + 1
        z0 = mu0 / x0
        r = z0 - np.dot(M0, x0) - q0
        qn1 = (k + 1) * mu0 - np.dot(q0, x0)

        MM = np.hstack((M0, r.reshape(-1, 1)))
        MM = np.vstack((MM, np.append(-r, 0)))
        qq = np.append(q0, qn1)
        x_init = np.append(x0, 1)
        z_init = np.append(z0, mu0)
        return MM, qq, x_init, z_init

    def _compute_objective(self):
        return np.dot(self.x, self.z) / len(self.x)

    def _compute_direction(self, x, z, M, gamma=0):
        mu = self._compute_objective()
        dx = np.linalg.solve(M + np.diag(z / x), gamma * mu / x - z)
        dz = gamma * mu / x - z - z * dx / x
        return dx, dz

    def solve(self, max_iter=100):
        while True:
            # Predictor step
            dx, dz = self._compute_direction(self.x, self.z, self.M)
            step_size = binary_search_step_size(self.x, self.z, dx, dz, 0.5)
            self.x += step_size * dx
            self.z += step_size * dz

            # Corrector step
            dx, dz = self._compute_direction(self.x, self.z, self.M, gamma=1)
            self.x += dx
            self.z += dz

            # compute the objective value
            mu = self._compute_objective()

            # increment counter by 1
            self.count += 1

            # print
            print("%sth iteration: step_size = %.5f, objective value = %.5f" % (self.count, step_size, mu))

            # conditions to terminate
            if mu < self.eps:
                self.primal_vars = self.x[self.m:self.m + self.n] / self.x[-2]
                self.dual_vars = self.x[:self.m] / self.x[-2]
                print("Optimal value = %.3f" % np.dot(c, self.primal_vars))
                print("Optimal value (dual) = %.3f" % np.dot(b, self.dual_vars))
                break

            if self.count > max_iter:
                print("Optimal value is not found")
                break


if __name__ == "__main__":
    c = np.array([4, 3, 5])
    A = np.array([[2, 2, -1], [2, -2, 3], [0, 2, -1]])
    b = np.array([6, 8, 4])
    print("A toy example")
    prob = SelfDualPrimalDualPathFollowing(c, A, b)
    prob.solve()
    x, y = prob.primal_vars, prob.dual_vars
    print(np.round(x, 3))
    print(np.round(y, 3))
    print()

    c = np.array([150, 200, 300])
    A = np.array([[3, 1, 2], [1, 3, 0], [0, 2, 4]])
    b = np.array([60, 36, 48])
    print("Production problem")
    prob = SelfDualPrimalDualPathFollowing(c, A, b)
    prob.solve()
    x, y = prob.primal_vars, prob.dual_vars
    print(np.round(x, 3))
    print(np.round(y, 3))
    print()
