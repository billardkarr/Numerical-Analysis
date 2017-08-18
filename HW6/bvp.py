from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as sla


def bvp_solve(p, q, f, x, u_a, u_b):
    n, dx = len(x), x[1] - x[0]

    D2 = sps.diags([1, -2, 1] / dx ** 2, offsets=[0, 1, 2], shape=(n - 2, n))
    D1 = sps.diags([-1, 1] / (2 * dx), offsets=[0, 2], shape=(n - 2, n))
    D0 = sps.diags([1], offsets=[1], shape=(n - 2, n))
    P = sps.diags([p(x[1:-1])], [0])
    Q = sps.diags([q(x[1:-1])], [0])

    left_bv = sps.diags([1], offsets=[0], shape=(1,n))
    right_bv = sps.diags([1], offsets=[n-1], shape=(1,n))

    A = sps.csr_matrix(sps.vstack([left_bv, D2 + P * D1 + Q * D0, right_bv]))
    b = f(x)
    b[0], b[-1] = u_a, u_b
    return sla.spsolve(A, b), A


def case(n):
    if n == 0:
        def u_true(x):
            return np.exp(-4 * x) / 3 + 2 * np.exp(2 * x) / 3

        def p(x):
            return 2 * (x * 0 + 1)

        def q(x):
            return -8 * (x * 0 + 1)

        def f(x):
            return x * 0

        a, b = 0, 1

    if n == 1:
        def u_true(x):
            return np.sin(np.log(x))

        def p(x):
            return 2 / x

        def q(x):
            return -2 / (x ** 2)

        def f(x):
            return (np.cos(np.log(x)) - 3 * np.sin(np.log(x))) / (x ** 2)

        a, b = 1, 2

    if n == 2:
        def u_true(x):
            return -np.sin(x) + 3 * np.cos(x)

        def p(x):
            return -1 * (0 * x + 1)

        def q(x):
            return -2 * (0 * x + 1)

        def f(x):
            return -8 * np.cos(x) + 6 * np.sin(x)

        a, b = 0, np.pi / 2

    return u_true, p, q, f, a, b

conds = np.empty(9)
for i in xrange(3):
    print "Case %d:" % (i + 1)
    print
    u_true, p, q, f, a, b = case(i)
    u_a, u_b = u_true(a), u_true(b)
    mesh = np.linspace(a, b, 2 ** 20)

    H = np.array([(b - a) * 2 ** -k for k in range(3, 20)])
    n = [2 ** k for k in range(3, 20)]
    errors = np.zeros(len(n))
    EOC = np.zeros(len(n) - 1)
    for k in xrange(len(n)):
        x = np.linspace(a, b, n[k])
        u, A = bvp_solve(p, q, f, x, u_a, u_b)
        if k < len(conds) and i == 0:
            conds[k] = np.linalg.cond(A.todense())
        errors[k] = max(abs(u - u_true(x)))
        if k > 0:
            EOC[k - 1] = np.log(errors[k] / errors[k - 1]) / \
                np.log(H[k] / H[k - 1])
            print "for n = %g, error: %g, EOC: %g" % (n[k], errors[k], EOC[k - 1])
        else:
            print "for n = %g, error: %g" % (n[k], errors[k])
    print

    plt.loglog(H, errors, 'o', label="Case %d" % (i+1))

plt.loglog(H, H ** 2, 'k-', label="$f(h) = h^2$")
plt.grid()
plt.legend(loc=4)
plt.title("Max error vs. $h$")
plt.xlabel('$h$')
plt.ylabel('$ \max_{a \leq x_i \leq b} | u_h (x_i) - u(x_i)| $')    
#plt.savefig("errors.png", dpi=200)
plt.show()


plt.grid()
plt.title("Condition number of $A$ vs. size $n$ of $A$")
plt.xlabel('$n =$ len($A$)')
plt.ylabel('cond($A$)')    
plt.loglog(n[:len(conds)], conds, 'o')
#plt.savefig("conds.png", dpi=200)
plt.show()
