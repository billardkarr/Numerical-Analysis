from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12)

a, b = 0, 1

def f(x):
	return 4/(1 + x**2)


def midpoint_rule(f,a,b,n):
	mesh = np.linspace(a,b,n)
	lengths = mesh[1:] - mesh[:-1]
	midpts = (mesh[1:] + mesh[:-1])/2
	return np.dot(f(midpts),lengths)


def trapezoid_rule(f,a,b,n):
	mesh = np.linspace(a,b,n)
	lengths = mesh[1:] - mesh[:-1]
	return np.dot((f(mesh[:-1])+f(mesh[1:]))/2,lengths)


def simpsons_rule(f,a,b,n):
	mesh = np.linspace(a,b,n)
	lengths = mesh[1:] - mesh[:-1]
	left, right = mesh[:-1], mesh[1:]
	mid = (right+left)/2
	simp = (f(left) + 4*f(mid) + f(right))/6
	return np.dot(simp,lengths)


def monte_carlo_method(f,a,b,n):
	mesh = a + b*np.random.rand(n)
	return sum(f(mesh))*(b-a)/n

names = ["Midpoint rule","Trapezoid rule","Simpson's rule","Monte Carlo method"]
image_names = ["midpoint","trapz","simpson","monte_carlo"]
letters = ['a','b','c','d']
methods = [midpoint_rule,trapezoid_rule,simpsons_rule,monte_carlo_method]
N = [19,19,5,20]

approx = [np.zeros(N[i]) for i in range(4)]

pi = 3.14159265358979323846264338327950288419716939937510582097494459230781

for i in range(len(methods)):
	print "For %s:" % names[i]
	rel_errs = np.zeros(N[i])
	H = [(b - a)/(2**(n+2)) for n in range(N[i])]
	#approx = np.zeros(N[i])
	print "Approximations of pi:"
	for n in xrange(N[i]):
		EOC = 'none'
		method = methods[i]
		approx[i][n] = method(f,a,b,2**(n+2))
		rel_errs[n] = abs(approx[i][n] - pi)/pi
		if n > 0:
			EOC = np.log(rel_errs[n]/rel_errs[n-1])/np.log(H[n]/H[n-1])
		print "h = 2**-%d, pi = %.15f, EOC = %s" % (n+2,approx[i][n],str(EOC))

	
	m,intercept = np.polyfit(np.log(H), np.log(rel_errs), 1)
	x = np.linspace(np.log(H[0]),np.log(H[-1]),2)
	y = m*x + intercept

	plt.loglog(np.exp(x),np.exp(y),'k-')
	plt.loglog(H,rel_errs,'o',label='%s, order of convergence = %g' % (names[i],m))
	print

fig = plt.gcf()
plt.title('2(ii): Relative error in approximations of $\pi$')
plt.ylabel('Relative Error')
plt.xlabel('$h$')
plt.legend(loc=3,prop={'size':10})
#plt.savefig("pi.png",dpi=200)
plt.show()