from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.special as sp


class CompositeLegendreDiscretization:

    """A discrete function space on a 1D domain consisting of multiple
    subintervals, each of which is discretized as a polynomial space of
    maximum degree *order*. (see constructor arguments)

    There are :attr:`nintervals` * :attr:`npoints` degrees of freedom
    representing a function on the domain. On each subinterval, the
    function is represented by its function values at Gauss-Legendre
    nodes (see :func:`scipy.special.legendre`) mapped into that
    subinterval.

    .. note::

        While the external interface of this discretization
        is exclusively in terms of vectors, it may be practical
        to internally reshape (see :func:`numpy.reshape`) these
        vectors into 2D arrays of shape *(nintervals, npoints)*.

    The object has the following attributes:

    .. attribute:: intervals

        The *intervals* constructor argument.

    .. attribute:: nintervals

        Number of subintervals in the discretization.

    .. attribute:: npoints

        Number of points on each interval. Equals *order+1*.

    .. attributes:: nodes

        A vector of ``(nintervals*npoints)`` node locations, consisting of
        Gauss-Legendre nodes that are linearly (or, to be technically correct,
        affinely) mapped into each subinterval.

    """

    def __init__(self, intervals, order):
        """
        :arg intervals: determines the boundaries of the subintervals to
            be used. If this is ``[a,b,c]``, then there are two subintervals
            :math:`(a,b)` and :math:`(b,c)`.
            (and the overall domain is :math:`(a,c)`)

        :arg order: highest polynomial degree being used
        """

        self.intervals = intervals
        self.nintervals = len(intervals) - 1
        self.npoints = order + 1
        self.mid_pts = (self.intervals[1:] + self.intervals[:-1]) / 2
        self.scales = (self.intervals[1:] - self.intervals[:-1]) / 2
        self.sample_nodes = sp.legendre(self.npoints).weights[:, 0].real
        self.sample_weights = sp.legendre(self.npoints).weights[:, 1].real
        # taking real part because sometimes sp.legendre is returning
        # complex numbers with zero imaginary part and displaying a warning

        nodes = (self.mid_pts + np.outer(self.sample_nodes, self.scales)).T
        self.nodes = np.reshape(nodes, -1)
        weights = np.outer(self.scales, self.sample_weights)
        self.weights = np.reshape(weights, -1)

        monos = np.array([self.sample_nodes ** k for k in range(self.npoints)])
        integrals = np.array([(self.sample_nodes ** (k + 1) - (-1) ** (k + 1)) / (k + 1)
                             for k in range(self.npoints)])
        self.spec_int_mat = la.solve(monos, integrals)

    def integral(self, f):
        r"""Use Gauss-Legendre quadrature on each subinterval to approximate
        and return the value of

        .. math::

            \int_a^b f(x) dx

        where :math:`a` and :math:`b` are the left-hand and right-hand edges of
        the computational domain.

        :arg f: the function to be integrated, given as function values
            at :attr:`nodes`
        """
        return np.dot(f, self.weights)

    def left_indefinite_integral(self, f):
        r"""Use a spectral integration matrix on each subinterval to
        approximate and return the value of

        .. math::

            g(x) = \int_a^x f(x) dx

        at :attr:`nodes`, where :math:`a` is the left-hand edge of the
        computational domain.

        The computational cost of this routine is linear in
        the number of degrees of freedom.

        :arg f: the function to be integrated, given as function values
            at :attr:`nodes`
        """
        d = self.npoints
        n = self.nintervals
        f = f.reshape((n, d))
        weights = self.weights.reshape((n, d))
        integrals = np.cumsum(np.einsum('ij,ij->i', f, weights))
        integrals = np.roll(integrals, 1)
        integrals[0] = 0
        indef = np.dot((f.T * self.scales).T, self.spec_int_mat)
        indef = (integrals + indef.T).T
        indef = np.reshape(indef, -1)
        return indef

    def right_indefinite_integral(self, f):
        r"""Use a spectral integration matrix on each subinterval to
        approximate and return the value of

        .. math::

            g(x) = \int_x^b f(x) dx

        at :attr:`nodes`, where :math:`b` is the left-hand edge of the
        computational domain.

        The computational cost of this routine is linear in
        the number of degrees of freedom.

        :arg f: the function to be integrated, given as function values
            at :attr:`nodes`
        """

        return self.integral(f) - self.left_indefinite_integral(f)
