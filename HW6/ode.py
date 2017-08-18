from __future__ import division
import numpy as np
import matplotlib.pyplot as plt



def runge_kutta(f, h, t0, tf, v0):
    n = int((tf - t0) / h) + 1
    v = np.zeros((n, 2))
    v[0] = v0
    t = np.linspace(t0, tf, n)
    for k in xrange(n - 1):
        k1 = f(v[k])
        k2 = f(v[k] + (h / 2) * k1)
        k3 = f(v[k] + (h / 2) * k2)
        k4 = f(v[k] + h * k3)
        v[k + 1] = v[k] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return v.T, t


def leapfrog(f, h, t0, tf, v0):
    n = int((tf - t0) / h) + 1
    v = np.zeros((n, 2))
    v[0] = v0
    t = np.linspace(t0, tf, n)
    v[1] = v[0] + h * f(v[0])
    v[1] = v[0] + (h / 2) * (f(v[0]) + f(v[1]))
    for k in xrange(n - 2):
        v[k + 2] = v[k] + 2 * h * f(v[k + 1])
    return v.T, t


def f(x):
    A = np.array([[0, 1], [-1, 0]])
    return np.dot(A, x)

T = 2 * np.pi  # period length
v0 = np.array([1, 0])  # [ u(0), u'(0) ]

methods = [runge_kutta, leapfrog]
methodnames = ['Runge-Kutta', 'Leapfrog']

for i in range(len(methods)):
    method = methods[i]
    print methodnames[i]
    print

    N = 1  # number of periods
    t0, tf = 0, N * T
    H = np.array([T * (2 ** -k) for k in range(3, 12)])
    errors = np.empty(len(H))
    EOC = np.empty(len(H))

    for j in range(len(H)):
        h = H[j]
        X, mesh = method(f, h, t0, tf, v0)
        errors[j] = max(abs(np.cos(mesh) - X[0]))
        if j > 0:
            EOC[j] = np.log(errors[j] / errors[j - 1]) / \
                np.log(H[j] / H[j - 1])
        print "For h = %g, max error = %g, EOC = %g" % (h, errors[j], EOC[j])

    print

    N = 400  # number of periods
    t0, tf = 0, N * T
    H = [T / 15, T / 20, T / 25, T / 30]

    for j in range(len(H)):
        h = H[j]
        print "Computing for h = %g" % h
        X, mesh = method(f, h, t0, tf, v0)
        E = (X[0] ** 2 + X[1] ** 2) / 2
        plt.plot(mesh, E, '-', label='$E_h(t)$ for $h = %g$' % h)
    print

    plt.legend(loc=3)
    plt.grid()
    plt.title("Energy for approximations to $ u'' = -u $, %s method" %
              methodnames[i])
    plt.ylim([0, 0.6])
    plt.xlim([0, N * T])
    plt.xlabel('$t$')
    plt.ylabel('$E(t)$')
    #plt.savefig("energy_%s.png" % methodnames[i].lower(),dpi=200)
    plt.show()

N = 400  # number of periods
t0, tf = 0, N * T
h = T / 20
mesh = np.linspace(t0, tf, 1e5)
plt.plot(mesh, np.cos(mesh), '-', label='$u(t) = \cos(t)$')

for i in range(len(methods)):
    method = methods[i]
    X, mesh = method(f, h, t0, tf, v0)
    plt.plot(mesh, X[0], '-', label='%s' % methodnames[i])

plt.legend(loc=2)
plt.grid()
plt.title("Approximate solutions to $ u'' = -u$, h = %g" % h)
plt.ylim([-1.5, 1.5])
plt.xlim([300 * T, 310 * T])
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
#plt.savefig("ode.png", dpi=200)
plt.show()