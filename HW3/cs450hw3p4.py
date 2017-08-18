# Bill Karr's Code for Assignment 3, Problem 4

from __future__ import division
import numpy as np
    
def Newton_iteration(f,df,x0,order,tol):
    maxit = 100
    iteration = 0
    x = x0
    
    while abs(f(x)) >= tol and iteration < maxit:
        iteration = iteration+1
        x = x - (order*f(x))/df(x)    
    
    return x, iteration
    
def find_rate(f,df,x0,x,order,tol):
    maxit = 100
    iteration = 0
    error = abs(x - x0)
    perror = 1
    pperror = 1/np.sqrt(error)
    r = 2

    while abs(f(x0)) >= tol and perror > 0 and pperror > 0 and abs(np.log(perror/pperror)) > 0 and iteration < maxit:
        if iteration > 1:    
            r = np.log(error/perror)/np.log(perror/pperror)
            print "rate = ", r
        
        iteration = iteration+1
        x0 = x0 - (order*f(x0))/df(x0)
                
        pperror = perror
        perror = error
        error = abs(x - x0)
        
    return r
    
tol = 1e-64
    
# Part (b)(i)

print "For f(x) = x^2 -1, x0 = 10^6"
print
    
def f1(x):
    return x**2 - 1   
def df1(x):
    return 2*x
    
x0 = 10**6
x,N = Newton_iteration(f1,df1,x0,1,tol)
rate = find_rate(f1,df1,x0,1,1,tol=1e-200)

print 
print "x* =",x
print "f'(x*) =", df1(x) 
print "Finished after", N,"iterations."

# Part (b)(ii)

print
print "For f(x) = (x^2 - 1)^4, x0 = 1000"
print
    
def f2(x):
    return (x**2 - 1)**4
def df2(x):
    return 8*x*((x**2-1)**3)
    
x0 = 1000
x, N = Newton_iteration(f2,df2,x0,4,tol)
rate = find_rate(f2,df2,x0,1,4,tol=1e-200)

print 
print "x* =",x
print "f'(x*) =", df2(x) 
print "Finished after", N,"iterations."

# Part (b)(iii)

print
print "For f(x) = x - cos(x), x0 = 1"
print

def f3(x):
    return x - np.cos(x) 
def df3(x):
    return 1 + np.sin(x)
    
x0 = 1
x, N = Newton_iteration(f3,df3,x0,1,tol)
rate = find_rate(f3,df3,x0,x,1,tol=1e-200)

print 
print "x* =",x
print "f'(x*) =", df3(x) 
print "Finished after", N,"iterations."