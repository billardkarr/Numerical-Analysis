from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def case(n, c):
    if n == 0:
        def g(x):
            return np.exp( -100*( x - 0.5)**2 )

    if n == 1:
        def g(x):
            result = 0 * x.copy()
            result[(x % 2) >= 0.5] = 1
            result[(x % 2) > 1] = 0
            return result

    def u(x, t):
        return g(x - c * t)

    return g, u


def runge_kutta(f, dx, t, u0):
        # Solves du/dt = f(u) for u with u(x,0) = u0
    n = len(t)
    dt = t[1] - t[0]
    u = np.zeros((n, len(u0)))
    u[0] = u0
    for k in xrange(n - 1):
        k1 = f(u[k], dx)
        k2 = f(u[k] + (dt / 2) * k1, dx)
        k3 = f(u[k] + (dt / 2) * k2, dx)
        k4 = f(u[k] + dt * k3, dx)
        u[k + 1] = u[k] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return u


def eulers_method(f, dx, t, u0):
    n = len(t)
    dt = t[1] - t[0]
    u = np.zeros((n, len(u0)))
    u[0] = u0
    for k in xrange(n - 1):
        u[k + 1] = u[k] + dt * f(u[k], dx)
    return u


def centered_diff(u, dx):
    result = -c * (np.roll(u[:-1], -1) - np.roll(u[:-1], 1)) / (2 * dx)
    return np.append(result, result[0])


def upwind_diff(u, dx):
    if c >= 0:
        result = -c * (u[:-1] - np.roll(u[:-1], 1)) / dx
        return np.append(result, result[0])
    else:
        result = -c * (np.roll(u[:-1], -1) - u[:-1]) / dx
        return np.append(result, result[0])


def two_norm(u, x):
    return np.sqrt(spi.simps(abs(u) ** 2, x))


def one_norm(u, x):
    return spi.simps(abs(u), x)


def infinity_norm(u):
    return np.array([max(abs(u[k])) for k in range(len(u))])


x = np.linspace(0, 2, 400)
dx = x[1] - x[0]
c = 1

steppers = [eulers_method, runge_kutta]
stepper_names = ["Euler's Method", "Runge-Kutta"]
diff_methods = [upwind_diff, centered_diff]
diff_names = ["Upwind Differences", "Centered Differences"]

for n in range(2):
    g, u_true = case(n, c)
    f, subs = plt.subplots(2, 2)
    plt.title('Case %d' % (n + 1))
    for i in range(2):
        stepper = steppers[i]
        for j in range(2):
            diff_method = diff_methods[j]

            dt = 1 * dx / abs(c)
            if dt == 0:
                dt = dx

            t = np.linspace(0, 10, int(10 / dt) + 1)

            u0 = g(x)
            u = stepper(diff_method, dx, t, u0)
            mesh, times = np.meshgrid(x, t)
            sol = u_true(mesh, times)

            L_infty_errors = infinity_norm(u - sol) / infinity_norm(sol)
            L_one_errors = one_norm(u - sol, x) / one_norm(sol, x)
            L_two_errors = two_norm(u - sol, x) / two_norm(sol, x)

            subs[i, j].set_title(r'%s, %s' % (stepper_names[i], diff_names[j]))
            subs[i, j].set_xlabel(r'$t$')
            subs[i, j].set_ylabel(r'$ \| u(\cdot,t) \|_p $')
            if max(abs(L_one_errors)) > 1:
                subs[i, j].semilogy(t, L_infty_errors, label=r"$ p = \infty $")
                subs[i, j].semilogy(t, L_two_errors, label=r"$ p = 2 $")
                subs[i, j].semilogy(t, L_one_errors, label=r"$ p = 1 $")
            else:
                subs[i, j].plot(t, L_infty_errors, label=r"$ p = \infty $")
                subs[i, j].plot(t, L_two_errors, label=r"$ p = 2 $")
                subs[i, j].plot(t, L_one_errors, label=r"$ p = 1 $")
            subs[i, j].legend(loc=4)
            subs[i, j].grid()

    # f.set_size_inches(12, 8)
    # plt.savefig('pde_errs_case%d.png' % (n+1), dpi=100)
    plt.show()

stable_combos = [[eulers_method, upwind_diff],
                 [runge_kutta, upwind_diff],
                 [runge_kutta, centered_diff]]

function_names = ['Sinusoidal wave', 'Tringular wave ', 'Rectangular wave']


for j in range(2):
    [stepper, diff_method] = stable_combos[j]
    fig = plt.figure()
    for n in range(3):
        g, u_true = case(n, c)

        dt = (1 * dx / abs(c))
        if dt == 0:
            dt = dx

        t = np.linspace(0, 2, int(2 / dt) + 1)

        u0 = g(x)
        u = stepper(diff_method, dx, t, u0)
        mesh, times = np.meshgrid(x, t)

        ax = fig.add_subplot(1, 3, n + 1, projection='3d')

        #subs[0,n].subplot(111, projection="3d")
        ax.plot_surface(mesh, times, u)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$u$')
        ax.set_title(function_names[n])

    # fig.set_size_inches(14, 5)
    # plt.savefig('pde_solution_combo%d.png' % (j+1), dpi=200)
    plt.show()
