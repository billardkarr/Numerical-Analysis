from __future__ import division
import numpy as np

A = np.array([[1,1,1],[3,2,-4],[4,5,7]])
b = np.array([1,2,3])
print "A =", A
print "b =", b

def gaussElim(A,b):
    # upper triangulate
    for col in range(len(A[0])):
        for row in range(col+1,len(A)):
            multiplier = A[row][col]/A[col][col]
            for j in range(col,len(A[0])):
                A[row][j] = A[row][j] - multiplier*A[col][j]
                b[row] = b[row] - multiplier*b[col]
                
    x = np.array([0]*len(A[0]))

    # backsolve
    for col in reversed(range(len(A))):
        x[col] = b[col]/A[col][col]
        for row in range(col-1):
            b[row] = b[row] - A[row][col]*x[col]
    return x
    
x = gaussElim(A,b)

print "x =", x

# check the answer
bstar = [np.dot(A[row],x) for row in range(len(b))]
diff = [bstar[row]-b[row] for row in range(len(b))]

print "Ax =", bstar
print "relative Error = ", np.linalg.norm(diff)/np.linalg.norm(b)
    