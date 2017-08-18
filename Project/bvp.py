from __future__ import division
import numpy as np
import numpy.linalg as la
from gmres import my_gmres_e


def apply_kernel(a, b, mesh, kernel, density):
    r"""Return a vector *F* corresponding to

    .. math::

        F(x) = \int_a^b k(x,z) \phi(z) dz

    evaluated at all points of *mesh* using the
    trapezoidal rule.

    :arg mesh: A 1D array of nodes in the interval :math:`[a, b]`,
          with the first equal to :math:`a` and the last equal to :math:`b`.
    :arg kernel: two-argument vectorized callable ``kernel(tgt, source)``
        that evaluates
    :arg density: Values of the density at the nodes of *mesh*.
    """
    def trapz(f, mesh):
        return np.dot((f[1:] + f[:-1]) / 2, mesh[1:] - mesh[:-1])

    temp = np.zeros(len(mesh))
    for i in xrange(len(mesh)):
        left = trapz(kernel(mesh[i], mesh[:i + 1], -1) * density[:i + 1], mesh[:i + 1])
        right = trapz(kernel(mesh[i], mesh[i:], +1) * density[i:], mesh[i:])
        temp[i] = left + right
    return temp


def solve_bvp(mesh, p, q, r, ua, ub):
    r"""Solve the boundary value problem

    .. math::

        u''+p(x)u'+q(x)u=r(x),\quad u(a) = u_a,\quad u(b) =u_b

    on *mesh*. Return a vector corresponding to the solution *u*.
    Uses :func:`apply_fredholm_kernel`.

    :arg mesh: A 1D array of nodes in the interval :math:`[a, b]`,
          with the first equal to :math:`a` and the last equal to :math:`b`.
    :arg p, q, r: Functions that accept a vector *x* and
      evaluate the functions $p$, $q$, $r$ at the nodes in *x*.
    """
    x = mesh
    a, b = mesh[0], mesh[-1]
    L = b - a

    def tau(x):
        return 1 - (x - a) / L

    def R(x):
        return -(q(x) * (tau(x) * ua + (1 - tau(x)) * ub) + p(x) * (ub - ua) / L - r(x))

    def K(x, z, sign):
        temp = np.empty(len(z))
        if sign == -1:
            temp[z <= x] = (p(x) + (x - b) * q(x)) * (z[z <= x] - a) / L
            temp[z > x] = (p(x) + (x - a) * q(x)) * (z[z > x] - b) / L
        if sign == +1:
            temp[z < x] = (p(x) + (x - b) * q(x)) * (z[z < x] - a) / L
            temp[z >= x] = (p(x) + (x - a) * q(x)) * (z[z >= x] - b) / L
        return temp

    def K2(x, z, sign):
        temp = np.empty(len(z))
        if sign == -1:
            temp[z <= x] = (x - b) * (z[z <= x] - a) / L
            temp[z > x] = (x - a) * (z[z > x] - b) / L
        if sign == +1:
            temp[z < x] = (x - b) * (z[z < x] - a) / L
            temp[z >= x] = (x - a) * (z[z >= x] - b) / L
        return temp

    def A_func(phi):
        return phi + apply_kernel(a, b, mesh, K, phi)

    print "Starting GMRES with n = %d:" % len(x)
    phi, its = my_gmres_e(A_func, R(x))
    print "solution found."
    print
    u = tau(x) * ua + (1 - tau(x)) * ub + apply_kernel(a, b, mesh, K2, phi)
    return u
