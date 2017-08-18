# Bill Karr's Code for Assignment 3, Problem 3

from __future__ import division
import numpy as np
np.random.seed(12)

# Part B

def reduce_to_hessenberg(A):
    A = A.astype(float)
    n = len(A)
    Q = np.eye(n)
    U = A.copy()  
    
    # perform Householder transformations
    for k in range(n-2):
        v = np.zeros(n)
        v[k+1:n] = U.T[k][k+1:n]
        alpha = -np.copysign(1,v[k+1])*np.linalg.norm(v)
        v[k+1] = v[k+1] - alpha
        beta = np.dot(v,v)
        if beta == 0:
            k = k+1
        else:
            H = np.eye(n) - (2/beta)*np.outer(v,v)
            Q = np.dot(H,Q)
            U = np.dot(np.dot(H,U),H)
            U[np.abs(U) < 1e-15] = 0

    return Q, U
    
# Part C
    
n = 10
A = np.random.rand(n,n)
Q,U = reduce_to_hessenberg(A)

print "Part C:"
print
print "|QQ.T - I| =", np.linalg.norm(np.dot(Q,Q.T) - np.eye(n))
print "|Q.T*U*Q - A|/|A| =", np.linalg.norm(np.dot(np.dot(Q.T,U),Q) - A)/np.linalg.norm(A)

# Part D

print
print "Part D"
print

n = 10
B = np.random.rand(n,n)
A = B + B.T
Q,U = reduce_to_hessenberg(A)

print "|Q.T*U*Q - A|/|A| =", np.linalg.norm(np.dot(np.dot(Q.T,U),Q) - A)/np.linalg.norm(A)
print "|QQ.T - I| =", np.linalg.norm(np.dot(Q,Q.T) - np.eye(n)), "<-- when this is small, Q is orthogonal"
print "|U.T - U|/|U| =", np.linalg.norm(U.T - U)/np.linalg.norm(U), "<-- when this is small, U is symmetric"