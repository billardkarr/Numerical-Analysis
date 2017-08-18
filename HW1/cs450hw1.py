# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:53:46 2014

@author: billkarr
"""
from scipy.special import jv

# PART A
# Here we are computing the degree to which Python's Bessel function
# computer obeys the relevant recurrance relation.

print "PART A"
print " "

for i in xrange(0,2):
    print "For n =", i
    print "J_n(20) =", jv(i,20)
for i in xrange(2,51):
    print "For n =", i, "J_n(20) =", jv(i,20)
    print "Relation Error =", (jv(i,20) - (2*(i-1)*jv(i-1,20)/20 - jv(i-2,20)))/jv(i,20)


# PART B
# Here we use Python's estimates of the first two and the recursio
# relation to compute the values of the higher Bessel functions.
# We measure the relative error between the actual Bessel function
# and the value computed using the recursion relation.

print " "
print "PART B"
print " "

besselRecursiveEst = [jv(0,20),jv(1,20)];
for i in xrange(2,51):
    besselRecursiveEst.append((2*(i-1)*besselRecursiveEst[i - 1])/20. - besselRecursiveEst[i - 2])
    
for i in xrange(0,51):
    print "For n =", i
    print "Bessl =", jv(i,20)
    print "RcEst =", besselRecursiveEst[i]
    print "RelEr =", (jv(i,20) - besselRecursiveEst[i])/jv(i,20)