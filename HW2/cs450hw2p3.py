from __future__ import division
import numpy as np
np.random.seed(12)

# Bill Karr's Code for Problem 3 in Problem set 2

# Problem 3, Part A
# QR factorization of a matrix A using modified Gram-Schmidt procedure

def GramSchmidtQR(A):
    A = A.astype(float)
    Q = A.copy()
    R = np.zeros((len(A.T),len(A.T)))
    # perform Gram-Schmidt on columns of A    
    for k in range(len(A.T)):
        R[k][k] = np.linalg.norm(Q.T[k])
        if R[k,k] == 0:
            raise ValueError("Matrix does not have full rank.")
        Q.T[k] = Q.T[k]/R[k][k]
        for j in range(k+1,len(A.T)):
            R[k][j] = np.dot(A.T[j],Q.T[k])
            Q.T[j] = Q.T[j] - R[k][j]*Q.T[k] 
    return Q, R
    
# Problem 3, Part B
# QR factorization of a matrix A using Householder transformations

def HouseholderQR(A):
    A = A.astype(float)
    m = len(A)
    n = len(A.T)
    Q = np.eye(m)
    R = A.copy()
    # perform Householder transformations
    for k in range(n):
        v = np.zeros(m)
        v[k:m] = R.T[k][k:m]
        alpha = -np.copysign(1,v[k])*np.linalg.norm(v)
        v[k] = v[k] - alpha
        beta = np.dot(v,v)
        if beta == 0:
            k = k+1
        else:
            H = np.eye(m) - (2/beta)*np.outer(v,v)
            Q = np.dot(Q,H)
            R = np.dot(H,R)
            R[np.abs(R) < 1e-15] = 0

    return Q, R
    
# Problem 3, Part C
print
print "Results for Problem 3, Part C:"
print

for [m,n] in [[5,5],[10,10],[100,80]]:
    
    A = np.random.rand(m,n)

    Q, R = GramSchmidtQR(A)
    print "size(A) =", len(A), "by", len(A.T)
    # print "Gram-Schmidt: relative error between I and Q.T * Q =", np.linalg.norm(np.eye(len(Q.T)) - np.dot(Q.T,Q))
    print "relative error between A and QR using Gram-Schmidt =", np.linalg.norm(A - np.dot(Q,R))/np.linalg.norm(A)
    
    Q1, R1 = HouseholderQR(A)
    # print "Householder: relative error between I and Q.T * Q =", np.linalg.norm(np.eye(len(Q1.T)) - np.dot(Q1.T,Q1))
    print "relative error between A and QR using Householder =", np.linalg.norm(A - np.dot(Q1,R1))/np.linalg.norm(A)
    
    print "cond(A) =", np.linalg.cond(A)
    print 


# Problem 3, Part D
print
print "Results for Problem 3, Part D:"
print

import matplotlib.pyplot as plt

# import data from file
price = np.genfromtxt("Price_of_Gasoline.txt", delimiter="\n")
price = np.array(price)
T = np.array(range(1,len(price)+1))

def PolyFit(method,price,T,d):
    B = np.vander(T,d+1)
    Q,R = method(B)
    b = np.dot(Q.T,price)
    coeff = np.linalg.solve(R[0:d+1][:],b[0:d+1])
    residual = np.linalg.norm(np.dot(B,coeff)-price)/np.linalg.norm(price)
    return coeff, residual

letters = ["a", "b", "c", "d", "e", "f"]
polys = ["a","a*T + b","a*T^2 + b*T + c","a*T^3 + b*T^2 + c*T + d","a*T^4 + b*T^3 + c*T^2 + d*T + e","a*T^5 + b*T^4 + c*T^3 + d*T^2 + e*T + f"]

print "Using Gram-Schmidt QR factorization:"
print 
plt.plot(T,price,'o',label='Original data',markersize=5)
for d in range(1,6):
    print "Degree of fitted polynomial =", d
    print
    
    # compute coefficients
    coeff, residual = PolyFit(GramSchmidtQR,price,T,d)
    print " P(T) =", polys[d]
    for i in range(len(coeff)):
        print letters[i],"=", coeff[i]
    print "relative residual =", residual
    print
    
    # generate plot for current polynomial
    plt.plot(T,sum([coeff[i]*T**(d - i) for i in range(d+1)]))
#plt.legend()
plt.title('3(d): polynomials fitted to data using Gram-Schmidt QR')
plt.xlabel('time')
plt.ylabel('price ($)')
plt.show()

print "Using Householder QR factorization:"
print
plt.plot(T,price,'o',label='Original data',markersize=5)
for d in range(1,6):
    print "Degree of fitted polynomial =", d
    print
    coeff, residual = PolyFit(HouseholderQR,price,T,d)
    print " P(T) =", polys[d]
    for i in range(len(coeff)):
        print letters[i],"=", coeff[i]
    print "relative residual =", residual
    print
    
    plt.plot(T,sum([coeff[i]*T**(d - i) for i in range(d+1)]))
#plt.legend()
plt.title('3(d): polynomials fitted to data using Householder QR')
plt.xlabel('time')
plt.ylabel('price ($)')
plt.show()


print "Using numpy least squares:"
print
plt.plot(T,price,'o',label='Original data',markersize=5)
for d in range(1,6):
    print "Degree of polynomial to fit =", d
    print
    
    B = np.vander(T,d+1)
    coeff = np.linalg.lstsq(B,price)[0]
    residual = np.linalg.norm(np.dot(B,coeff)-price)/np.linalg.norm(price)
    print " P(T) =", polys[d]
    for i in range(len(coeff)):
        print letters[i],"=", coeff[i]
    print "relative residual =", residual
    print
    
    plt.plot(T,sum([coeff[i]*T**(d - i) for i in range(d+1)]))
#plt.legend()
plt.title('3(d): polynomials fitted to data using numpy.lstsq')
plt.xlabel('time')
plt.ylabel('price ($)')
plt.show()