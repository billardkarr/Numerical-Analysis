from __future__ import division
import numpy as np
from fast_bvp import solve_bvp


a = -4
b = 5

def test_poisson(discr):
    def u_true(x):
        return np.sin(5*x)*np.exp(x)

    def r(x):
        return 10*np.exp(x)*np.cos(5*x)-24*np.exp(x)*np.sin(5*x)

    def p(x):
        return 0

    def q(x):
        return 0

    u = solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(discr.nodes)
    err = u-u_true_val

    return np.sqrt(discr.integral(err**2))


def test_with_q(discr):
    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return 0

    def q(x):
        return 1/(x+7)

    def r(x):
        return np.sin(5*x**2)/(x+7)-100*x**2*np.sin(5*x**2)+10*np.cos(5*x**2)

    u = solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(discr.nodes)
    err = u-u_true_val

    return np.sqrt(discr.integral(err**2))


def test_with_p(discr):
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

    u = solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(discr.nodes)
    err = u-u_true_val

    return np.sqrt(discr.integral(err**2))


def test_full_enchilada(discr):
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

    u = solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(discr.nodes)
    err = u-u_true_val

    return np.sqrt(discr.integral(err**2))


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
    from legendre_discr import CompositeLegendreDiscretization

    for test in [test_poisson, test_with_p, test_with_q, test_full_enchilada]:
        for order in [3, 5, 7]:
            print "----------------------------------------------"
            print "%s -- order %d" % (test.__name__, order)
            print "----------------------------------------------"

            def get_error(n):
                intervals = np.linspace(0, 1, n, endpoint=True) * (b-a) + a
                discr = CompositeLegendreDiscretization(intervals, order)
                return 1/len(intervals), test(discr)

            assert estimate_order(get_error, [50, 100]) >= order - 0.5
