from __future__ import division
import numpy as np
import numpy.linalg as la
from gmres import my_gmres_e
from legendre_discr import CompositeLegendreDiscretization


def apply_kernel(discr, fl, gl, fr, gr, density):
    """
    :arg discr: an instance of
        :class:`legendre_discr.CompositeLegendreDiscretization`
    :arg fl,gl,fr,gr: functions of a single argument
    """
    x = discr.nodes
    Gl = discr.left_indefinite_integral(gl(x) * density)
    Gr = discr.right_indefinite_integral(gr(x) * density)
    result = fl(x) * Gl + fr(x) * Gr
    return result


def solve_bvp(discr, p, q, r, ua, ub):
    """
    :arg discr: an instance of
        :class:`legendre_discr.CompositeLegendreDiscretization`
    """
    a, b = discr.intervals[0], discr.intervals[-1]
    L = b - a

    def tau(x):
        return 1 - (x - a) / L

    def R(x):
        return -(q(x) * (tau(x) * ua + (1 - tau(x)) * ub) + p(x) * (ub - ua) / L - r(x))

    def fl(x):
        return (p(x) + (x - b) * q(x)) / L

    def fr(x):
        return (p(x) + (x - a) * q(x)) / L

    def gl(x):
        return x - a

    def gr(x):
        return x - b

    def fl2(x):
        return (x - b) / L

    def fr2(x):
        return (x - a) / L

    def A_func(phi):
        return phi + apply_kernel(discr, fl, gl, fr, gr, phi)

    x = discr.nodes
    print "Starting GMRES:"
    phi, its = my_gmres_e(A_func, R(x))
    print "solution found."
    print

    u = tau(x) * ua + (1 - tau(x)) * ub + \
        apply_kernel(discr, fl2, gl, fr2, gr, phi)

    return u
