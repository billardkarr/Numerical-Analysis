from __future__ import division
import numpy as np
from fast_bvp import solve_bvp
from legendre_discr import CompositeLegendreDiscretization
import matplotlib.pyplot as plt
import time

a = -4
b = 5


def test_full_enchilada(discr):
    def u_true(x):
        return np.sin(5 * x ** 2)

    def p(x):
        return np.cos(x)

    def q(x):
        return 1 / (x + 7)

    def r(x):
        return (np.sin(5 * x ** 2) / (x + 7)
                - 100 * x ** 2 * np.sin(5 * x ** 2)
                + 10 * x * np.cos(x) * np.cos(5 * x ** 2) + 10 * np.cos(5 * x ** 2))

    u = solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b))

    u_true_val = u_true(discr.nodes)
    err = u - u_true_val

    return np.sqrt(discr.integral(err ** 2))

order = 5
min_power = 10
max_power = 17
m = 2 ** min_power
M = 2 ** max_power

sizes = [2 ** k for k in range(min_power, max_power + 1)]
powers = range(min_power, max_power + 1)
times = np.zeros(len(sizes))

for i in range(len(sizes)):
    intervals = np.linspace(0, 1, sizes[i], endpoint=True) * (b - a) + a
    discr = CompositeLegendreDiscretization(intervals, order)
    t0 = time.time()
    err = test_full_enchilada(discr)
    times[i] = time.time() - t0
    print "For n = 2^%d, time elapsed: %g seconds, L^2 error: %g" % (powers[i], times[i], err)
    print

rate, b = np.polyfit(np.log(sizes), np.log(times), 1)
x = np.linspace(np.log(m), np.log(M), 2)
y = rate * x + b

plt.loglog(np.exp(x), np.exp(y), 'r-')
plt.loglog(sizes, times, 'ko')
plt.xlim([m / 1.1, M * 1.1])
plt.title(
    r'Computational time of fast BVP Solver, $ t(n) \propto \, n^{%g} $.' % rate)
plt.xlabel('Number of subintervals')
plt.ylabel('Elapsed time (seconds)')
plt.show()
fig = plt.gcf()
#plt.savefig('timing.png', dpi=200)
