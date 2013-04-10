from numpy import *
from os import system

set_printoptions(precision=4)

system('clear')
Xtrue = array([[2.0, 3.0, 15.4]])

Xpoints = array([[1.8, 3.3, 16.4],[1.9, 3.5, 14.4],[2.23, 4, 15.7],[1.5, 2.6, 12.4],[1.5, 2.6, 18.4],[1.5, 2.6, 13.4]])
# Xpoints = array([[2.0, 3.0, 15.4],[2.0, 3.0, 15.4],[2.0, 3.0, 15.4],[2.0, 3.0, 15.4],[2.0, 3.0, 15.4],[2.0, 3.0, 15.4]])

Q = 0.5*eye(3)

n, N = Xpoints.shape
w = 1.0/float(n)

P = Q
for x in Xpoints:
	error = x - Xtrue 
	P    = P + w*dot(error.T,error)
S = linalg.cholesky(P)
print "Covariance matrix"
print P
print "\nSquare root matrix with cholesky"
print S
print "\nTest S*S' - P = 0"
print dot(S,S.T)-P

# doing the same with QR factorization
i = 0
for x in Xpoints:
	error = sqrt(w)*(x - Xtrue)
	if i==0:
		A = error.T
	else:
		A = hstack((A,error.T))
	i += 1
A = hstack((A,linalg.cholesky(Q)))
print "\nA matrix before factorization is "+str(A.shape)
print A
q,r = linalg.qr(A.T,mode='full')

print "\nq matrix orthogonal is"
print q
print dot(q,q.T)
print "\nr matrix is"
print r.T