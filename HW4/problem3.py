# Bill Karr's Code for Problem 3 on Assignment 4
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
import numpy.linalg as la
 
# Part A
def f(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
def grad_f(x):
    return np.array([2*(2*x[0]*(x[0]**2+x[1]-11)+x[0]+x[1]**2-7),2*(x[0]**2+2*x[1]*(x[0]+x[1]**2-7)+x[1]-11)])
    	
def Hess_f(x):
    return np.array([[-42 + 12*x[0]**2 + 4*x[1], 4*(x[0] + x[1])],[4*(x[0] + x[1]),-26 + 4*x[0] + 12*x[1]**2]])

def steep(x):
    s = -1*grad_f(x)
    return getAlpha(f,grad_f,x,s)*s

def newton(x):
    return la.solve(Hess_f(x),-grad_f(x))

def dampNewton(x):
    s = newton(x)
    return getAlpha(f,grad_f,x,s)*s
    
def getAlpha(f,grad_f,x,s):
    alpha = spo.line_search(f,grad_f,x,s)[0]
    if alpha is None:
        alpha = 1
    return alpha

def get_iterates(method,starter,maxit=500,tol=1e-14):
    x = np.array(starter)
    iterates = x
    dx = method(x)
    while la.norm(dx)/la.norm(x) >= tol:
        x = x + dx
        iterates = np.row_stack((iterates,x))
        dx = method(x)
        if len(iterates) == maxit:
            break
    print "Number of iterations:", len(iterates)
    return np.transpose(iterates), x
    
starters = [[2, 2],[2,-1],[-2,2],[-2,-2]]
methods, methodnames = [steep,newton,dampNewton],["Steepest Descent","Newton's Method","Damped Newton's Method"]
    
for method in methods:
    for starter in starters:
        iterates, final = get_iterates(method,starter)
        plt.plot(iterates[0],iterates[1],'k-',label='path taken')
        plt.plot(final[0],final[1],'g*',label='terminal point')
    xmesh, ymesh = np.meshgrid(np.linspace(-4,4,801),np.linspace(-4,4,801))
    fmesh = f((xmesh, ymesh))
    plt.axis([-4,4,-4,4])
    plt.title("Problem 3(a): " + methodnames[methods.index(method)])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.contour(xmesh, ymesh, fmesh,label='f(x)')
    fig = plt.gcf()
    plt.savefig('p3afig'+str(methods.index(method)+1)+'.png',dpi=200)
    plt.show()

# Part B
def g(t,x):
    return x[0] + x[1]*t + x[2]*(t**2) + x[3]*np.exp(x[4]*t)
    
def grad_g(t,x):
    return np.array([1,t,t**2,np.exp(t*x[4]),x[3]*t*np.exp(t*x[4])])

def r(x):
    return np.array([Y[i] - g(T[i],x) for i in range(len(T))])
    
def jacobian_r(x):
    return np.array([-grad_g(T[i],x) for i in range(len(T))] )

def gaussNewton(x):
    return la.lstsq(jacobian_r(x),-r(x))[0]
    
Y,T = [20,51.58,68.73,75.46,74.36,67.09,54.73,37.98,17.28], np.linspace(0,2,9)
starters = [[0,0,0,0,1],[1,0,0,0,0],[1,0,0,1,0]]

for starter in starters:
    iterates, final = get_iterates(gaussNewton,starter)
    print "Final x=", final
    tmesh = np.linspace(-0.1,2.1,1e3)
    gmesh = g(tmesh,final)
    plt.plot(tmesh, gmesh,'-',label='model function')
    plt.plot(T,Y,'k.',label='data')
    fig = plt.gcf()
    plt.title('Problem 3(b): Nonlinear data fitting, $x_0 ='+str(starter)+'$, '+str(len(iterates[0]))+' iterations')
    plt.xlabel('$t$')
    plt.ylabel('$y$')
    plt.axis([min(tmesh),max(tmesh), 10,80])
    plt.legend(prop={'size':10})
    plt.savefig("p3bfig" + str(starters.index(starter)+1) + ".png",dpi=200)
    plt.show()