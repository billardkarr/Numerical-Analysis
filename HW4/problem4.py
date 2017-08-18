# Bill Karr's Code for Problem 4 on Assignment 4
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import scipy.special as ss

def lagrange(x_vals,j,x): # Lagrange basis polynomials
    return reduce(op.mul, [(x - x_vals[i])/(x_vals[j] - x_vals[i]) for i in xrange(len(x_vals)) if i != j])

def lam(x_vals,x): # Lambda_T(x)
    return sum( abs(lagrange(x_vals,i,x)) for i in xrange(len(x_vals)))

def getnodes(nodetype,size):
    if nodetype == "Equispaced nodes":
        return np.linspace(-1,1,size)
    if nodetype == "Chebyshev nodes":
        return np.cos((2*np.pi*np.linspace(1,n,n)+1)/(2*n))
    if nodetype == "Gauss-Legendre nodes":
        return ss.legendre(n).weights[:,0]
        
nodenames = ["Equispaced nodes","Chebyshev nodes","Gauss-Legendre nodes"]

lebesgues = np.zeros((3,5))
for nodetype in nodenames:
    for n in range(5,30,5):
        nodes = getnodes(nodetype,n)
        xmesh = np.linspace(-1,1,2000)
        lmesh = lam(nodes,xmesh)
        lebesgues[nodenames.index(nodetype),range(5,30,5).index(n)] = max(lmesh)
    plt.semilogy(range(5,30,5),lebesgues[nodenames.index(nodetype)],'.',label=nodetype)
    
plt.xlim([4,26])
plt.title('Problem 4: Lebesgue constant estimation')
plt.xlabel('number of nodes')
plt.ylabel('Lebesgue constant')
plt.legend(loc=2,prop={'size':9})
fig = plt.gcf()
plt.savefig('p4.png',dpi=200)
plt.show()