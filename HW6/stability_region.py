from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.integrate as spi
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D

from cmath import exp, pi


def approximate_stability_region_1d(step_function, make_k, prec=1e-10):
    def is_stable(k):
        y = 1
        for i in range(100):
            if abs(y) > 2:
                return False
            y = step_function(y, i, 1, lambda t, y: k * y)
        return True

    def refine(stable, unstable):
        assert is_stable(make_k(stable))
        assert not is_stable(make_k(unstable))
        while abs(stable - unstable) > prec:
            mid = (stable + unstable) / 2
            if is_stable(make_k(mid)):
                stable = mid
            else:
                unstable = mid
        else:
            return stable

    mag = 1
    if is_stable(make_k(mag)):
        mag *= 2
        while is_stable(make_k(mag)):
            mag *= 2

            if mag > 2 ** 8:
                return mag
        return refine(mag / 2, mag)
    else:
        mag /= 2
        while not is_stable(make_k(mag)):
            mag /= 2

            if mag < prec:
                return mag
        return refine(mag, mag * 2)


def get_stability_region(center, stepper):
    def make_k(mag):
        return center + mag * exp(1j * angle)

    stab_boundary = []
    for angle in np.linspace(0, 2 * np.pi, 200):
        stable_mag = approximate_stability_region_1d(stepper, make_k)
        stab_boundary.append(make_k(stable_mag))

    stab_boundary = np.array(stab_boundary)
    return stab_boundary


def get_eigs(method, n, dt):
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    I = np.eye(n)
    A = np.empty((n, n))
    for i in range(n):
        A[i] = method(I[i], dx)
    return dt * la.eigvals(A)


def centered_diff(u, dx, c=1):
    result = -c * (np.roll(u[:-1], -1) - np.roll(u[:-1], 1)) / (2 * dx)
    return np.append(result, result[0])


def upwind_diff(u, dx, c=1):
    if c >= 0:
        result = -c * (u[:-1] - np.roll(u[:-1], 1)) / dx
        return np.append(result, result[0])
    else:
        result = -c * (np.roll(u[:-1], -1) - u[:-1]) / dx
        return np.append(result, result[0])


def fw_euler_step(y, t, h, f):
    return y + h * f(t, y)


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


c = 1
dt = np.array([[0.00249, 0.0001], [0.0035, 0.0069]])
dx = 1 / 400
C = c * dt / dx

print C

steppers = [fw_euler_step, rk4_step]
diff_method = [upwind_diff, centered_diff]

steppernames = list(['Forward Euler', 'Runge-Kutta'])
diff_names = list(['Upwind Differences', 'Centered Differences'])

euler = get_stability_region(-1, fw_euler_step)
rk4 = get_stability_region(-1, rk4_step)
regions = [euler, rk4]

f, subs = pt.subplots(2, 2, sharex='col', sharey='row')
for i in xrange(2):
    for j in xrange(2):
        eigs = get_eigs(diff_method[j], 400, dt[i, j])
        subs[i, j].set_title(r'%s, %s, $C \approx %g$' %
                             (steppernames[i], diff_names[j], C[i, j]))
        subs[i, j].set_aspect('equal')
        subs[i, j].grid()
        subs[i, j].plot(regions[i].real, regions[i].imag)
        subs[i, j].plot(eigs.real, eigs.imag, '.')
# f.set_size_inches(12, 8)
# pt.savefig('stability_regions.png', dpi=100)
pt.show()
