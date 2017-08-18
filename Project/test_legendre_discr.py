from __future__ import division

from legendre_discr import CompositeLegendreDiscretization
import numpy as np
import matplotlib.pyplot as pt


def get_left_int_error(n, order):
    a = 2
    b = 30
    intervals = np.linspace(0, 1, n, endpoint=True) ** 2 * (b-a) + a
    discr = CompositeLegendreDiscretization(intervals, order)

    x = discr.nodes

    assert abs(discr.integral(1+0*x) - (b-a)) < 1e-13

    alpha = 4
    from scipy.special import jv, jvp
    f = jvp(alpha, x)

    num_int_f = jv(alpha, a) + discr.left_indefinite_integral(f)
    int_f = jv(alpha, x)

    if 0:
        pt.plot(x.ravel(), num_int_f.ravel())
        pt.plot(x.ravel(), int_f.ravel())
        pt.show()

    L2_err = np.sqrt(discr.integral((num_int_f - int_f)**2))
    return 1/n, L2_err


def get_right_int_error(n, order):
    a = 2
    b = 30
    intervals = np.linspace(0, 1, n, endpoint=True) ** 2 * (b-a) + a
    discr = CompositeLegendreDiscretization(intervals, order)

    x = discr.nodes

    assert abs(discr.integral(1+0*x) - (b-a)) < 1e-13

    alpha = 4
    from scipy.special import jv, jvp
    f = jvp(alpha, x)

    num_int_f = jv(alpha, b) - discr.right_indefinite_integral(f)
    int_f = jv(alpha, x)

    if 0:
        pt.plot(x.ravel(), num_int_f.ravel())
        pt.plot(x.ravel(), int_f.ravel())
        pt.show()

    L2_err = np.sqrt(discr.integral((num_int_f - int_f)**2))
    return 1/n, L2_err


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
    for order in [2, 3, 5, 7]:
        print "---------------------------------"
        print "ORDER", order
        print "---------------------------------"
        assert (estimate_order(lambda n: get_left_int_error(n, order), [10, 30])
                >= order-0.5)
        assert (estimate_order(lambda n: get_right_int_error(n, order), [10, 30])
                >= order-0.5)
