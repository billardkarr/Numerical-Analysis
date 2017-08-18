from __future__ import division
import numpy as np

import matplotlib.pyplot as plt

def f(x):
	return np.sin(x)

def df(x):
	return np.cos(x)

def num_diff(f,a,b,h):
	n = int((b-a)/h)
	x = np.linspace(a,b,n)
	h = (b-a)/n
	return x, (- 3*f(x) + 4*f(x+h) - f(x+(2*h)))/(2*h)

a, b = -1, 1
k = np.array(range(3,21))
h = 2**(-k.astype(float))

errors = np.zeros(len(k))
for i in range(len(k)):
	x, ndf = num_diff(f,a,b,h[i])
	errors[i] = max(ndf - df(x))

EOC = np.log(errors[1:]/errors[:-1])/np.log(h[1:]/h[:-1])

print "Errors:", errors
print "EOC:", EOC

plt.loglog(h,errors,'o')
fig = plt.gcf()
plt.title('Problem 4(c): error of numerical differentiation, $ f(x) = \sin(x) $')
plt.xlabel('$h$')
plt.ylabel('Max error on $ [-1,1] $')
#plt.savefig("diff.png",dpi=200)
plt.show()