import numpy as np
from itertools import zip_longest
MEPS = 1.0e-10

def simplex(c, A, b):
    """
    This function solve a linear programming using Simplex method.
    The problem should be formulated in the inequality standard form such that
        maximize c^Tx
        subject to Ax <= b, x >= 0

    Args:
    c: np.array of shape (n,)
    A: np.array of shape (m, n)
    b: np.array of shape (m,)

    Outputs:
    primal and dual variables
    """
    nrows, ncols = A.shape # m = nrows, n = ncols
    AI = np.hstack((A, np.identity(nrows)))
    c0 = np.concatenate((c, np.zeros(nrows)))
    basis = [ncols + i for i in range(nrows)]
    nonbasis = [j for j in range(ncols)]

    while True:
        y = np.linalg.solve(AI[:, basis].T, c0[basis])
        cc = c0[nonbasis] - np.dot(y, AI[:, nonbasis])

        if np.all(cc <= MEPS):
            x = np.zeros(nrows + ncols)
            x[basis] = np.linalg.solve(AI[:, basis], b)
            primal_vars, dual_vars = x[:nrows], y[:nrows]
            print("Optimal")
            print("Optimal value = %.3f" % np.dot(c, primal_vars))
            print("-" * 30)
            for i, (p_var, d_var) in enumerate(zip_longest(primal_vars, dual_vars)):
                print("x_%s = %.3f, y_%s = %.3f" % (i+1, p_var, i, d_var))
            print("-" * 30)
            return primal_vars, dual_vars

        entering_id = np.argmax(cc) # Dantzig's rule
        d = np.linalg.solve(AI[:, basis], AI[:, nonbasis[entering_id]])

        if np.all(d <= MEPS):
            print("Unbounded")
            break

        bb = np.linalg.solve(AI[:, basis], b)
        ratio = np.repeat(np.inf, nrows)
        ratio[d > MEPS] = [bb_i / d_i for (bb_i, d_i) in zip(bb, d) if d_i > MEPS]
        leaving_id = np.argmin(ratio)
        nonbasis[entering_id], basis[leaving_id] = basis[leaving_id], nonbasis[entering_id]

if __name__ == "__main__":
    A = np.array([[2, 2, -1], [2, -2, 3], [0, 2, -1]])
    c = np.array([4, 3, 5])
    b = np.array([6, 8, 4])

    print("A toy example")
    p_vars, d_vars = simplex(c, A, b)
    print()

    A = np.array([[3, 1, 2], [1, 3, 0], [0, 2, 4]])
    c = np.array([150, 200, 300])
    b = np.array([60, 36, 48])
    print("Production problem")
    simplex(c, A, b)