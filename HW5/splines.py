from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
np.random.seed(25)


def cubic_spline(T,Y):

	def poly_dk(coeff,t,k):
		dkT = np.array([ np.prod(range(i+1-k,i+1))*(t**(i-k)) for i in range(k,len(coeff)) ]).T
		return np.dot(coeff[k:],dkT)

	def spline_eqns(coeffs,T):
		coeffs = coeffs.reshape((len(T)-1,-1))
		nsplines = len(coeffs)
		left = np.array([[poly_dk(coeffs[i],T[i],k) for i in range(nsplines)] for k in range(3)])
		right = np.array([[poly_dk(coeffs[i],T[i+1],k) for i in range(nsplines)] for k in range(3)])
		boundaries = np.concatenate(left[1:,1:] - right[1:,:-1])
		ends = np.array([left[-1,0],right[-1,-1]])

		output = np.append(left[0,:],right[0,:])
		output = np.append(output,boundaries)
		output = np.append(output,ends)
		return output

	def build_spline_matrix(T):
		nsplines = len(T)-1
		num_eqns = 4*nsplines
		matrix = np.empty((num_eqns,num_eqns))
		for i in range(num_eqns):
			E = np.zeros(num_eqns)
			E[i] = 1
			matrix.T[i] = spline_eqns(E,T)
		return matrix
	
	spline_matrix = build_spline_matrix(T)
	b = np.zeros((N-1)*4)
	b[:N-1] = Y[:-1]
	b[N-1:2*(N-1)] = Y[1:]
	coeffs = la.solve(spline_matrix,b).reshape((N-1,-1))

	def final_interpolant(t):
		for i in range(N-1):
			if t >= T[i] and t <= T[i+1]:
				return poly_dk(coeffs[i],t,0)

	return np.vectorize(final_interpolant)


N = 6
T_data = np.sort(np.random.rand(N))
Y_data = np.random.rand(N)
F = cubic_spline(T_data,Y_data)

mesh = np.linspace(T_data[0],T_data[-1],1e4)
plt.plot(mesh,F(mesh),'-',label='spline interpolant')
plt.plot(T_data,Y_data,'o',label='data')
fig = plt.gcf()
plt.title('Problem 1(c): Cubic Spline Interpolation')
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.legend()
#plt.savefig("spline.png",dpi=200)
plt.show()