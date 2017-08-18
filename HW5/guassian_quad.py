from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt

def f(x):
	return np.exp(x)

def g(x):
	return abs(x)

def get_weights(nodes,a,b,tol=1e-15):
	V = np.array([nodes**i for i in xrange(len(nodes)) ])
	b = np.array([ (b**(i+1) - a**(i+1))/(i+1) for i in xrange(len(nodes)) ])
	if 1/la.cond(V) > tol:
		return la.solve(V,b)
	else:
		return scipy.special.legendre(len(nodes)).weights[:, 1]

def guass_quad(f,a,b,n):
	sample_nodes = sp.special.legendre(n).weights[:, 0]
	nodes = (a+b)/2 + sample_nodes*(b-a)/2
	weights = get_weights(nodes,a,b)
	return np.dot(f(nodes),weights)

N = 100
a,b = -1,1
functions = [f,g]
integrals = [np.exp(1)-np.exp(-1),1]
errors = [np.empty(N), np.empty(N)]
orders = 1+np.array(range(N))

funcs = ['f(x) = exp(x)','g(x) = |x|']
tex_functions = ['$ f(x) = e^x $','$ g(x) = |x| $']

for i in range(2):
	#print "f(x) = ", functions[i]
	errors = np.zeros(N)
	for j in range(N):
		integral = guass_quad(functions[i],a,b,orders[j])
		errors[j] = abs(integral-integrals[i])
		#print "For n = %d, error = %g" % (orders[j],errors[j])

	H = 1/orders
	EOC = np.log(errors[1:]/errors[:-1])/np.log(H[1:]/H[:-1])
	print "EOC values for %s:" % funcs[i]
	print EOC
	print

	plt.loglog(H,errors,'o')
	fig = plt.gcf()
	plt.title('Problem 3(c): Guassian quadrature on $ [-1,1] $, %s' % tex_functions[i])
	plt.xlabel('$1/n$ where $n$ is number of Gauss-Legendre nodes')
	plt.ylabel('Error')
	#plt.savefig("gauss%d.png" % i,dpi=200)
	plt.show()


