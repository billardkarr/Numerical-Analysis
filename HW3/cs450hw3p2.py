# Bill Karr's Code for Assignment 3, Problem 2

from __future__ import division
import numpy as np
np.random.seed(12)

# Part B

def Lanczos(A):
    n = len(A)
    Q = np.zeros((n,n+1))
    H = np.zeros((n,n))
    beta = 0
    x0 = np.random.randn(n)
    Q.T[1] = x0/np.linalg.norm(x0)
    for k in range(1,n):
        u = np.dot(A,Q.T[k])
        H[k-1][k-1] = np.dot(Q.T[k],u)
        u = u - beta*Q.T[k-1] - H[k-1][k-1]*Q.T[k]
        beta = np.linalg.norm(u)
        H[k-1][k] = beta
        H[k][k-1] = beta
        if beta == 0:
            raise ValueError('Matrix is reducible.')
        else:
            Q.T[k+1] = u/beta
    H[n-1][n-1] = np.dot(Q.T[n],np.dot(A,Q.T[n]))
    return Q[:,1:n+1],H
        
n = 25
B = np.random.randn(n,n)
A = B + B.T
Q,H = Lanczos(A.copy())

print "Part B:"
print "size(A) =", n, "by", n
print "|QQ.T - I| =", np.linalg.norm(np.dot(Q,Q.T) - np.eye(n))
print "|Q.T*A*Q - H|/|A| =", np.linalg.norm(np.dot(np.dot(Q.T,A),Q) - H)/np.linalg.norm(A)

print

# Part C
import matplotlib.pyplot as plt
print "Part C:"

n = 32
B = np.random.randn(n,n)
Q,R = np.linalg.qr(B)
D = np.diag(np.array(range(1,n+1)))
A = np.dot(np.dot(Q,D),Q.T)

Q,H = Lanczos(A.copy())

Ritz = np.array([[H[0][0]],[1]]).T
for i in range(1,len(H)+1):
    new_Ritz = np.array([np.linalg.eigvals(H[0:i,0:i]),i*np.ones(i)]).T
    Ritz = np.concatenate([Ritz,new_Ritz])   
Ritz = Ritz.T
    
plt.plot(Ritz[0],Ritz[1],'.')
plt.gca().set_aspect("equal")
plt.title('2(c): Ritz values after k iterations')
plt.xlabel('Ritz values')
plt.ylabel('k')

fig = plt.gcf()
fig.set_size_inches(4,4)
plt.savefig('p2fig1.png',dpi=200)
plt.show()