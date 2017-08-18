from __future__ import division
import numpy as np
from scipy import linalg
np.random.seed(12)

# Bill Karr's Code for Problem 4 in Problem set 2

# Problem 4, Part A
print
print "Part A: inverse iteration for computing eigenvalues and eigenvectors"
print

def invIteration(A,x,shift):
    relDiff = 1
    iteration = 0
    while relDiff > 10**(-12) and iteration < 10**4:
        iteration = iteration + 1
        y = np.linalg.solve(A - shift*np.eye(len(A)),x)
        y = y/y[np.argmax(np.absolute(y))]
        x = x*np.copysign(1,y[np.argmax(np.absolute(y))])
        relDiff = np.linalg.norm(x - y)/np.linalg.norm(x)
        x = y
    if iteration == 10**4:
        raise ValueError("Not converging quickly...")
    else:
        y = y/np.linalg.norm(y)
        lam = np.dot(y,np.dot(A,y))
        return y, lam, iteration
    
A = np.array([[6,2,1],[2,3,1],[1,1,1]])

x = np.random.rand(10,len(A))

print "Running inverse iteration..."
print

lam = np.zeros(10)
for i in range(10):
    v, lam[i], iteration = invIteration(A,x[i],2)
    print "Trial", i+1
    print "Approximate eigenvalue =", lam[i]
    print "Approximate eigenvector =", v
    print "Number of iterations =", iteration
    
# Problem 4, Part B
print
print "Part B: Using Rayleigh quotient iteration for computing eigenvalues and eigenvectors"
print

def RQI(A,x):
    relDiff = 1
    iteration = 0
    while relDiff > 10**(-12) and iteration < 10**4:
        iteration = iteration + 1
        sigma = np.dot(x,np.dot(A,x))/np.dot(x,x)
        y = np.linalg.solve(A - sigma*np.eye(len(A)),x)
        y = y/y[np.argmax(np.absolute(y))]
        x = x*np.copysign(1,y[np.argmax(np.absolute(y))])
        relDiff = np.linalg.norm(x - y)/np.linalg.norm(x)
        # print "relative difference =", relDiff
        x = y
    if iteration == 10**4:
        raise ValueError("Not converging quickly...")
    else:
        y = y/np.linalg.norm(y)
        lam = np.dot(y,np.dot(A,y))
        # print "iterations =", iteration
        return y, lam, iteration
    
print "Running Rayleigh Quotient iteration..."
print
    
lam = np.zeros(10)
for i in range(10):
    v, lam[i], iteration = RQI(A,x[i])
    print "Trial", i+1
    print "Approximate eigenvalue =", lam[i]
    print "Approximate eigenvector =", v
    print "Number of iterations =", iteration
    
# Problem 4, Part D
print
print "Part D: Comparing iterative methods to actual eigenvectors and eigenvalues"
print

E,V = linalg.eig(A)
# print "eigenvalues of A =", E
# print "eigenvectors of E =", V
print "Using inverse iteration:"
print
for i in range(10):
    print "Trial", i+1
    print "Starting vector:", x[i]
    v, lam[i], iteration = invIteration(A,x[i],2)
    k = np.argmin(np.abs(E - lam[i]))
    correctEV = E[k] # grab closest eigenvalue to approximate value
    V[k] = np.copysign(1,V[k][np.argmax(np.absolute(V[k]))])*V[k] # normalize sign of V
    print "Number of iterations:", iteration
    print
    
    print "Approximate eigenvalue:", lam[i]
    print "Actual eigenvalue:", correctEV
    relErrorLam = np.abs(correctEV - lam[i])/np.abs(correctEV) 
    print "Relative Error in eigenvalue:", relErrorLam
    print
    
    print "Approximate eigenvector:", v
    print "Actual eigenvector:", V[k]
    relErrorVec = np.linalg.norm(v - V[k])/np.linalg.norm(V[k])
    print "Relative Error in eigenvector:", relErrorVec
    print

print
print "Using Rayleigh quotient iteration:"
print
for i in range(10):
    print "Trial", i+1
    print "Starting vector:", x[i]
    v, lam[i], iteration = RQI(A,x[i])
    k = np.argmin(np.abs(E - lam[i]))
    correctEV = E[k] # grab closest eigenvalue to approximate value
    V[k] = np.copysign(1,V[k][np.argmax(np.absolute(V[k]))])*V[k] # normalize sign of V
    print "Number of iterations:", iteration
    print
    
    print "Approximate eigenvalue:", lam[i]
    print "Actual eigenvalue:", correctEV
    relErrorLam = np.abs(correctEV - lam[i])/np.abs(correctEV) 
    print "Relative Error in eigenvalue:", relErrorLam
    print
    
    print "Approximate eigenvector:", v
    print "Actual eigenvector:", V[k]
    relErrorVec = np.linalg.norm(v - V[k])/np.linalg.norm(V[k])
    print "Relative Error in eigenvector:", relErrorVec
    print
    
    
# Problem 4, Part E
print
print "Part E"
print
starter = np.array([1, 4, 2])
print "Starting vector:", starter

print
print "Using inverse iteration:"
print
v, lam, iteration = invIteration(A,starter,2)
k = np.argmin(np.abs(E - lam))
correctEV = E[k] # grab closest eigenvalue to approximate value
V[k] = np.copysign(1,V[k][np.argmax(np.absolute(V[k]))])*V[k] # normalize sign of V
print "Number of iterations:", iteration
print

print "Approximate eigenvalue:", lam
print "Actual eigenvalue:", correctEV
relErrorLam = np.abs(correctEV - lam)/np.abs(correctEV) 
print "Relative Error in eigenvalue:", relErrorLam
print

print "Approximate eigenvector:", v
print "Actual eigenvector:", V[k]
relErrorVec = np.linalg.norm(v - V[k])/np.linalg.norm(V[k])
print "Relative Error in eigenvector:", relErrorVec
print

print
print "Using Rayleigh quotient iteration:"
print
v, lam, iteration = RQI(A,starter)
k = np.argmin(np.abs(E - lam))
correctEV = E[k] # grab closest eigenvalue to approximate value
V[k] = np.copysign(1,V[k][np.argmax(np.absolute(V[k]))])*V[k] # normalize sign of V
print "Number of iterations:", iteration
print

print "Approximate eigenvalue:", lam
print "Actual eigenvalue:", correctEV
relErrorLam = np.abs(correctEV - lam)/np.abs(correctEV) 
print "Relative Error in eigenvalue:", relErrorLam
print

print "Approximate eigenvector:", v
print "Actual eigenvector:", V[k]
relErrorVec = np.linalg.norm(v - V[k])/np.linalg.norm(V[k])
print "Relative Error in eigenvector:", relErrorVec
print