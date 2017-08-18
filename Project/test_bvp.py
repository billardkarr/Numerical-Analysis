from __future__ import division
import numpy as np
from bvp import solve_bvp


def test_poisson(n):
    a = -4
    b = 5

    def u_true(x):
        return np.sin(5*x)*np.exp(x)

    def r(x):
        return 10*np.exp(x)*np.cos(5*x)-24*np.exp(x)*np.sin(5*x)

    def p(x):
        return 0

    def q(x):
        return 0

    mesh = np.linspace(a, b, n)
    u = solve_bvp(mesh, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(mesh)
    err = u-u_true_val

    L2_err = np.sqrt(np.trapz(err**2, mesh))

    return mesh[1] - mesh[0], L2_err


def test_with_q(n):
    a = -4
    b = 5

    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return 0

    def q(x):
        return 1/(x+7)

    def r(x):
        return np.sin(5*x**2)/(x+7)-100*x**2*np.sin(5*x**2)+10*np.cos(5*x**2)

    mesh = np.linspace(a, b, n)
    u = solve_bvp(mesh, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(mesh)
    err = u-u_true_val

    L2_err = np.sqrt(np.trapz(err**2, mesh))

    return mesh[1] - mesh[0], L2_err


def test_with_p(n):
    a = -4
    b = 5

    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return np.cos(x)

    def q(x):
        return 0

    def r(x):
        return (
                - 100*x**2*np.sin(5*x**2)
                + 10*x*np.cos(x)*np.cos(5*x**2)+10*np.cos(5*x**2)
                )

    mesh = np.linspace(a, b, n)
    u = solve_bvp(mesh, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(mesh)
    err = u-u_true_val

    L2_err = np.sqrt(np.trapz(err**2, mesh))

    return mesh[1] - mesh[0], L2_err


def test_full_enchilada(n):
    a = -4
    b = 5

    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return np.cos(x)

    def q(x):
        return 1/(x+7)

    def r(x):
        return (np.sin(5*x**2)/(x+7)
                - 100*x**2*np.sin(5*x**2)
                + 10*x*np.cos(x)*np.cos(5*x**2)+10*np.cos(5*x**2))

    mesh = np.linspace(a, b, n)
    u = solve_bvp(mesh, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(mesh)
    err = u-u_true_val

    L2_err = np.sqrt(np.trapz(err**2, mesh))

    return mesh[1] - mesh[0], L2_err


def estimate_order(f, point_counts):
    n1, n2 = point_counts
    h1, err1 = f(n1)
    h2, err2 = f(n2)

    print "h=%g err=%g" % (h1, err1)
    print "h=%g err=%g" % (h2, err2)

    from math import log
    est_order = (log(err2/err1) / log(h2/h1))
    print "%s: EOC: %g" % (f.__name__, est_order)
    print

    return est_order


if __name__ == "__main__":
    estimate_order(test_poisson, [300, 3000])
    estimate_order(test_with_p, [300, 3000])
    estimate_order(test_with_q, [300, 3000])
    estimate_order(test_full_enchilada, [300, 3000])
