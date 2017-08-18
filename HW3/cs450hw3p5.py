# Bill Karr's Code for Assignment 3, Problem 5

from __future__ import division
import numpy as np
np.random.seed(12)

# Part A

def f(sphericals,x0):
    [r,theta,phi] = sphericals
    [x,y,z] = x0
    a = r*np.sin(theta)*np.cos(phi) - x
    b = r*np.sin(theta)*np.sin(phi) - y
    c = r*np.cos(theta) - z
    return np.array([a,b,c])
    
def J(sphericals):
    [r,theta,phi] = sphericals
    V1 = [np.sin(theta)*np.cos(phi), r*np.cos(theta)*np.cos(phi),-r*np.sin(theta)*np.sin(phi)]
    V2 = [np.sin(theta)*np.sin(phi), r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.cos(phi)]
    V3 = [np.cos(theta)            ,-r*np.sin(theta),             0]
    M = np.array([V1,V2,V3])
    return M

def newton(f, J, x0, tol=1e-12, maxit=500):
    iteration = 0
    x = [1,np.pi/2,np.pi/2]
    while np.linalg.norm(f(x,x0)) >= tol and iteration < maxit:
        iteration = iteration+1
        s = np.linalg.solve(J(x),f(x,x0))
        x = x - s
    return x, iteration
    
# Part B

def get_sphericals(a,b,c):
    x0 = [a,b,c]
    sphericals, N = newton(f, J, x0, tol=1e-12, maxit=500)
    [r,theta,phi] = sphericals
    print "Number of iterations of Newton's method:", N
    return r,theta,phi
    
# Part C 

def get_cartesian(sphericals):
    [r,theta,phi] = sphericals
    a = r*np.sin(theta)*np.cos(phi)
    b = r*np.sin(theta)*np.sin(phi)
    c = r*np.cos(theta)
    return np.array([a,b,c])

vectors = np.random.randn(10,3)

for i in range(10):
    print "Trial", i+1
    print
    x0 = vectors[i]
    [x,y,z] = x0
    r,theta,phi = get_sphericals(x,y,z)
    sphericals = np.array([r,theta,phi])
    X = get_cartesian(sphericals)
    residual = np.linalg.norm(X - x0)/np.linalg.norm(x0)
    r_true = np.sqrt(x**2 + y**2 + z**2)
    theta_true = np.arccos(y/r_true)
    phi_true = np.arctan2(y,x)
    true_sphericals = np.array([r_true,theta_true,phi_true])
    print "Cartesian vector =", x0
    print
    print "Output using Newton's method =", sphericals
    print " True spherical coordinates  =", true_sphericals
    print
    print "relative residual =", residual
    print "relative error =", np.linalg.norm(true_sphericals - sphericals)/np.linalg.norm(true_sphericals)
    print
    