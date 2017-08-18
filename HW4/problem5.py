# Bill Karr's Code for Problem 5 on Assignment 4
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

def vander(nodetype,polytype,n):
    return np.array([[polytype(j,nodetype(n)[i]) for i in range(n)] for j in range(n)])

def monoPoly(n,t):
        return t**n
         
def chebPoly(n,t):
    return np.cos(n*np.arccos(t))
   
def equiNodes(n):
    return np.linspace(-1,1,n)
    
def chebNodes(n):
    return np.cos((2*np.pi*np.linspace(1,n,n)+1)/(2*n))
    
nodetypes, polytypes = [equiNodes,chebNodes],[monoPoly,chebPoly]
nodenames, polynames = ['equispaced nodes','Chebyshev nodes'],['monomials','Chebyshev polynomials']
[m, M, delta] = [5,100,5]

for poly in [0,1]:
    for nodes in [0,1]:
        temp = [la.cond(vander(nodetypes[nodes],polytypes[poly],n)) for n in range(m,M+m,delta)]
        plt.semilogy(range(m,M+m,delta),temp,'.',label=nodenames[nodes]+', '+polynames[poly])
    
plt.xlim([m - 0.5*delta,M + 0.5*delta])
plt.title('Problem 5(c): Conditioning of Vandermonde matrices')
plt.xlabel('number of nodes $n$')
plt.ylabel('Condition number of $ V_{n} $')
plt.legend(prop={'size':7},bbox_to_anchor=(0.43, 0.4), loc=2, borderaxespad=0.)
fig = plt.gcf()
plt.savefig('p5.png',dpi=200)
plt.show()