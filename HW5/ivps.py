from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def forward_eulers(f,h,t0,tf,y0):
	n = int((tf-t0)/h) + 1
	y = np.zeros(n)
	y[0] = y0
	t = np.linspace(t0,tf,n)
	for k in xrange(n-1):
		y[k+1] = y[k] + h*f(t[k],y[k])
	return y,t


def backward_eulers(f,h,t0,tf,y0):
	
	def solve_quadratic(a,b,c,sign):
		if a == 0:
			return -c/b
		else:
			if sign == 1:
				return (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
			if sign == -1:
				return (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
	
	n = int((tf-t0)/h) + 1
	y = np.zeros(n)
	y[0] = y0
	t = np.linspace(t0,tf,n)
	for k in xrange(n-1):
		y[k+1] = solve_quadratic(200*h*t[k+1],1,-y[k],1)
	return y,t


def runge_kutta(f,h,t0,tf,y0):
	n = int((tf-t0)/h) + 1
	y = np.zeros(n)
	y[0] = y0
	t = np.linspace(t0,tf,n)
	for k in xrange(n-1):
		k1 = f(t[k],y[k])
		k2 = f(t[k]+h/2,y[k]+(h/2)*k1)
		k3 = f(t[k]+h/2,y[k]+(h/2)*k2)
		k4 = f(t[k]+h,y[k]+h*k3)
		y[k+1] = y[k] + h*(k1 + 2*k2 + 2*k3 + k4)/6
	return y,t


def f(t,y):
	return -200*t*(y**2)


def g(t):
	return 1/(1 + 100*(t**2))


t0, tf = 0, 1
m,M = 4,8
powers = m + np.array(range(M-m+1))
H = np.array([2**(-power) for power in powers])

mesh = np.linspace(t0,tf,1e5)
y_true = g(mesh)

names = ["Forward Euler method","Backward Euler method","4th Order Runge-Kutta"]
filenames = ["f","b","rk"]
methods = [forward_eulers,backward_eulers,runge_kutta]
errors = np.empty((len(methods),len(H)))
EOC = np.empty((len(methods),len(H)-1))

for method in range(len(methods)):
	plt.plot(mesh,y_true,'-',label="$y(t) = (1 + 100t^2)^{-1} $")
	for i in range(len(H)):
		sol, T = methods[method](f,H[i],t0,tf,g(t0))
		errors[method,i] = abs(sol[-1]-g(T[-1]))/abs(g(T[-1]))
		plt.plot(T,sol,'-',label="$y_h(t)$, $h = %g$" % H[i])
	EOC[method] = np.log(errors[method][1:]/errors[method][:-1])/np.log(H[1:]/H[:-1])

	fig = plt.gcf()
	plt.title('Problem 5: Numerical IVP solver, %s' % names[method])
	plt.xlabel('$t$')
	plt.ylabel('$y$')
	plt.xlim([t0,tf])
	plt.ylim([0,1.05])
	plt.legend(prop={'size':10})
	#plt.savefig("ivp_sol_%s.png" % filenames[method],dpi=200)
	plt.show()

	plt.loglog(H,errors[method],'o')
	fig = plt.gcf()
	plt.title('Problem 5: $| y(1) - y_h(1) |/|y(1)|$ vs. $h$, %s' % names[method])
	plt.xlabel('$h$')
	plt.ylabel('$| y(1) - y_h(1) |/|y(1)|$')
	#plt.savefig("ivp_err_%s.png" % filenames[method],dpi=200)
	plt.show()

print "h values:"
print H
print

for i in range(len(names)):
	print "EOC values for %s:" % names[i]
	print EOC[i]
	print