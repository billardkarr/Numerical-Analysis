# Bill Karr's Code for Problem 3 in Problem set 1

from __future__ import division

from scipy.special import jv
import numpy as np

# <codecell>

# Problem 3, Part A

bessel = np.array([jv(i,20) for i in range(0,51)])
diff = np.array([(bessel[i] - (2*(i-1)*bessel[i-1]/20 - bessel[i-2]) ) for i in range(2,51)])
relError = np.array([(bessel[i] - (2*(i-1)*bessel[i-1]/20 - bessel[i-2]) )/bessel[i] for i in range(2,51)])
    
print "PART A"
print "n & scipy approx & LHS - RHS & % error"
for i in range(0,49):
    print i+2, "&", bessel[i], "&", diff[i], "&", 100*abs(relError[i])

# <codecell>

# Problem 3, Part B

approxBessel = np.array([jv(0,20),jv(1,20)])

for i in range(2,51):
    approxBessel = np.append( approxBessel, 2*(i-1)*approxBessel[i-1]/20 - approxBessel[i-2] )
    
relError2 = np.array([ (approxBessel[i] - bessel[i])/bessel[i] for i in range(0,51) ])

print "PART B"
print "n & scipy approx & rec. approx & rel error"
for i in range(0,51):
    print i, "&", bessel[i], "&", approxBessel[i], "&", abs(relError2[i])


# <codecell>

# Problem 3, Part D

approxBessel2 = [jv(49,20),jv(50,20)]
approxBessel2 = np.array(approxBessel2)

for i in reversed(range(0,49)):
    approxBessel2 =  np.append( 2*(i+1)*approxBessel2[0]/20 - approxBessel2[1], approxBessel2 )
    # appending the next term onto the front of the array using the recursion relation
    
relError3 = np.array([ (approxBessel2[i] - bessel[i])/bessel[i] for i in range(0,51) ])

print "PART D"
print "n & scipy approx & rec. approx & rel error"
for i in range(0,51):
    print i, "&", bessel[i], "&", approxBessel2[i], "&", abs(relError3[i])



