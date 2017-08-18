from __future__ import division
import numpy as np
np.random.seed(12)

# Bill Karr's Code for Problem 1 in Problem set 2

# Problem 1, Part D

#create random SPD matrix
def rand_spd(n):
    A = np.random.rand(n,n)
    return np.dot(A,A.T)

#compute Cholesky factor L of an SPD matrix A
def getCholesky(A):
    L = np.zeros((len(A),len(A)))
    for i in range(len(A)):
        for j in range(i):
            L[i][j] = (A[i][j] - np.dot(L[i][0:j],L[j][0:j]))/L[j][j]
        L[i][i] = np.sqrt( A[i][i] - np.dot(L[i][0:i],L[i][0:i]) )
    return L

print "Problem 3, Part D: We compute 3 random 20x20 SPD matrices, compute their"
print "Cholesky factors, measure the relative error between LL.T and A, and print" 
print "the condition number of A."
print

n = 20 # size of matrix  
for i in [0,1,2]:
    A = rand_spd(n) # create random SPD matrix
    L = getCholesky(A) # obtain Cholesky factor using code above
    check = np.dot(L,L.T) # compute L*L^T which should equal A
    relErr = np.linalg.norm(check - A)/np.linalg.norm(A) # compute relative error
    print "For matrix", i+1,":"
    print "relative error =", relErr
    print "cond(A) =", np.linalg.cond(A)
    print

# Problem 1, Part E

print "Problem 3, Part E: We compute random SPD matrices A of size 5x5, 10x10, and"
print "100x100, compute their Cholesky factors L, and then compute the determinant"
print "of A by computing the determinant of L and squaring it, and the relative"
print "error between this value and numpy's determinant of A."
print

for n in [5,10,100]:
    A = rand_spd(n) # create random SPD matrix
    L = getCholesky(A) # obtain Cholesky factor using code above
    detChol = np.product([L[i][i] for i in range(len(A))])**2 # product of diagonal elements and squared since A = L*L.T
    detNP = np.linalg.det(A)
    print "n =", n
    print "numpy det(A) =", detNP
    print "det(L)^2 =", detChol
    print "relative error =", abs(detChol - detNP)/abs(detNP)
    print 
