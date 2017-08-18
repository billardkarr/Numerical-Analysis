from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


def my_gmres_d(A_func, b, tol=1e-10):
    """Solve Ax = b to an absolute residual norm of at most tol.

    Returns a tuple (x, num_iterations).

    Solution to Part 2(d) of the project assignment.
    """
    if la.norm(b) < tol:
        return 0 * b, 0
    if la.norm(A_func(b)) == 0:
        print "Warning: Krylov subspaces are trivial."
        return 0 * b, 0

    n = len(b)
    H = np.empty((1, 0))
    Q = np.empty((n, 1))
    Q[:, 0] = b / la.norm(b)
    x = b * np.dot(b, A_func(b)) / np.dot(A_func(b), A_func(b))
    y = np.empty((2, 1))
    vec = np.array([la.norm(b)])

    for k in xrange(n):
        print "performing iteration %d of GMRES..." % (k+1)
        H = np.column_stack((H, np.zeros(k + 1)))
        H = np.row_stack((H, np.zeros(k + 1)))
        u = A_func(Q[:, k])
        for j in xrange(k + 1):
            H[j, k] = np.dot(Q[:, j], u)
            u = u - H[j, k] * Q[:, j]
        H[k + 1, k] = la.norm(u)
        if H[k + 1, k] == 0:
            break
        Q = np.column_stack((Q, u / H[k + 1, k]))

        vec = np.append(vec, 0)
        y = la.lstsq(H, vec)[0]
        x = np.dot(Q[:, :-1], y)
        res = la.norm(b - A_func(x)) / la.norm(b)
        if res < tol:
            break

    if la.norm(b - A_func(x)) / la.norm(b) >= tol:
        print "Warning: tolerance was not met."
    return x, k + 1


def my_gmres_e(A_func, b, tol=1e-10):
    """Solve Ax = b to an absolute residual norm of at most tol.

    Returns a tuple (x, num_iterations).

    Solution to Part 2(e) of the project assignment.
    """
    if la.norm(b) < tol:
        print "Solution is trivial."
        return 0 * b, 0
    if la.norm(A_func(b)) == 0:
        print "Warning: Krylov subspaces are trivial."
        return 0 * b, 0

    n = len(b)
    H = np.empty((1, 0))
    Q = np.empty((n, 1))
    Q[:, 0] = b / la.norm(b)
    x = b * np.dot(b, A_func(b)) / np.dot(A_func(b), A_func(b))
    givens = []
    y = np.empty((2, 1))
    R = np.empty((1, 0))
    vec = np.array([la.norm(b)])

    for k in xrange(n):
        print "performing iteration %d of GMRES..." % (k+1)
        H = np.column_stack((H, np.zeros(k + 1)))
        H = np.row_stack((H, np.zeros(k + 1)))
        u = A_func(Q[:, k])
        for j in xrange(k + 1):
            H[j, k] = np.dot(Q[:, j], u)
            u = u - H[j, k] * Q[:, j]
        H[k + 1, k] = la.norm(u)
        if H[k + 1, k] == 0:
            break
        Q = np.column_stack((Q, u / H[k + 1, k]))

        vec = np.append(vec, 0)
        R = np.row_stack((R, np.zeros(k)))
        R = np.column_stack((R, H[:, -1]))
        # Apply previous Givens rotations to new column of H
        for i in xrange(k):
            R[i:i + 2, -1] = np.dot(givens[i], R[i:i + 2, -1])
        [c, s] = R[-2:, -1] / la.norm(R[-2:, -1])
        givens.append(np.array([[c, s], [-s, c]]))
        R[-2:, -1] = np.dot(givens[k], R[-2:, -1])
        vec[-2:] = np.dot(givens[k], vec[-2:])
        y = sla.solve_triangular(R[:-1, :], vec[:-1])
        res = abs(vec[-1]) / la.norm(b)
        if res < tol:
            x = np.dot(Q[:, :-1], y)
            break

    if la.norm(b - A_func(x)) / la.norm(b) >= tol:
        print "Warning: tolerance was not met."
    return x, k + 1


def test_gmres(gmres_func):
    n = 100
    eigvals = 1 + 10 ** np.linspace(1, -20, n)
    eigvecs = np.random.randn(n, n)
    A = np.dot(
        la.solve(eigvecs, np.diag(eigvals)),
        eigvecs)

    def A_func(x):
        return np.dot(A, x)

    x_true = np.random.randn(n)
    b = np.dot(A, x_true)
    x, num_it = gmres_func(A_func, b)

    print "converged after %d iterations" % num_it
    print "residual: %g" % (la.norm(np.dot(A, x) - b) / la.norm(b))
    print "error: %g" % (la.norm(x - x_true) / la.norm(x_true))


if __name__ == "__main__":
    print "----------------------------------------"
    print "part(d)"
    print "----------------------------------------"
    test_gmres(my_gmres_d)
    print "----------------------------------------"
    print "part(e)"
    print "----------------------------------------"
    test_gmres(my_gmres_e)
