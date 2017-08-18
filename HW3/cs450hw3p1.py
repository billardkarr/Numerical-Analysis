# Bill Karr's Code for Assignment 3, Problem 1

from __future__ import division
import numpy as np

def qr_iteration(A, tol):
    n = len(A)
    for i in range(n-1,0,-1):
        while np.linalg.norm(A[i-1,:i-1]) >= tol:
            sigma = A[i][i]
            Q,R = np.linalg.qr(A - sigma*np.eye(n,n))
            A = np.dot(R,Q) + sigma*np.eye(n,n)
    return np.diag(A)

tol = 1e-16

A_1 = np.array([[2,3,2],[10,3,4],[3,6,1]])
eigenvalues_1 = qr_iteration(A_1.copy(), tol)
print "Matrix ="
print A_1
print "Computed eigenvalues: ", eigenvalues_1
print "Actual eigenvalues: ", np.linalg.eigvals(A_1)

A_2 = np.array([[6,2,1],[2,3,1],[1,1,1]])
eigenvalues_2 = qr_iteration(A_2.copy(), tol)
print "Matrix ="
print A_2
print "Computed eigenvalues: ", eigenvalues_2
print "Actual eigenvalues: ", np.linalg.eigvals(A_2)