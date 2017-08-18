from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Part A
print
print "Part A"

dims = 2
npts = 3000

def make_data(dims,npts):
    np.random.seed(13)
    
    mix_mat = np.random.randn(dims,dims)
    mean = np.random.randn(dims)
    
    return np.dot(mix_mat,np.random.randn(dims,npts)) + mean[:,np.newaxis]

data = make_data(dims,npts)
plt.plot(data[0],data[1],'.')
plt.gca().set_aspect("equal")
plt.title('5(a): Raw data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Part B
print
print "Part B"

mean = np.mean(data,axis=1)
recentered_data = data - np.array([mean for i in range(npts)]).T
Y = recentered_data/np.sqrt(npts-1)
U, Sigma, V = np.linalg.svd(Y,full_matrices=False)
Pcompts = np.dot(U,np.diag(Sigma))

plt.plot(data[0],data[1],'.')
plt.arrow(mean[0],mean[1],Pcompts[0][0],Pcompts[1][0],zorder=10)
plt.arrow(mean[0],mean[1],Pcompts[0][1],Pcompts[1][1],zorder=10)
plt.gca().set_aspect("equal")
plt.title('5(c): Data with principle components')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Part C
print
print "Part C"
print

YY = np.dot(np.dot(U,np.diag(Sigma)),V)
relError = np.linalg.norm(Y - YY)/np.linalg.norm(Y)
print "Relative error between Y and U*Sigma*V.T =", relError

# Part D
print
print "Part D"
print

Sigma2 = Sigma
Sigma2[1] = 0
Y2 = np.dot(np.dot(U,np.diag(Sigma2)),V)

new_data = np.sqrt(npts-1)*Y2 + np.array([mean for i in range(npts)]).T
plt.plot(new_data[0],new_data[1],'.')
plt.arrow(mean[0],mean[1],Pcompts[0][0],Pcompts[1][0],zorder=10)
plt.arrow(mean[0],mean[1],Pcompts[0][1],Pcompts[1][1],zorder=10)
plt.gca().set_aspect("equal")
plt.title('5(d): Cleaned up data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()