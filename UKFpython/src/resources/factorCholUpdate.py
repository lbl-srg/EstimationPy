import numpy as np
from os import system

np.set_printoptions(precision=4)

system('clear')
Xtrue = np.array([[2.0, 3.0, 15.4]])

Xpoints = np.array([[1.8, 3.3, 16.4],[1.9, 3.5, 14.4],[2.23, 4, 15.7],[1.5, 2.6, 12.4],[1.5, 2.6, 18.4],[1.5, 2.6, 13.4]])
# Xpoints = np.array([[2.0, 3.0, 15.4],[2.0, 3.0, 15.4],[2.0, 3.0, 15.4],[2.0, 3.0, 15.4],[2.0, 3.0, 15.4],[2.0, 3.0, 15.4]])

Q = 0.5*np.eye(3)

n, N    = Xpoints.shape

Weights = np.zeros(n)
for i in range(n):
	if i==0:
		Weights[i] = -0.5
	else:
		Weights[i] = (1.0 - Weights[0])/np.float(n-1)

i = 0
P = Q
for x in Xpoints:
	error = x - Xtrue 
	P     = P + Weights[i]*np.dot(error.T,error)
	i    += 1
S = np.linalg.cholesky(P)

# doing the same with QR factorization + Cholesky Update
i = 0
for x in Xpoints:
	error = np.sqrt(Weights[i])*(x - Xtrue)
	if i==1:
		A = error.T
	elif i>1:
		A     = np.hstack((A,error.T))
	i    += 1
A = np.hstack((A,np.linalg.cholesky(Q)))

q,r = np.linalg.qr(A.T,mode='full')


# NOW START THE CHOLESKY UPDATE
print "\nStarting the cholesky update"
error0 = Xpoints[0,]-Xtrue

L = r.copy()
x = error0[0].copy()

absW  = np.abs(Weights[0])
signW = np.sign(Weights[0])
x     = signW*np.sqrt(absW)*x

print "\nCholupdate...\n"

for k in range(N):
	rr_arg    = L[k,k]**2 + signW*x[k]**2
	rr        = 0.0 if rr_arg < 0 else np.sqrt(rr_arg)
	c         = rr / L[k,k]
	s         = x[k] / L[k,k]
	L[k,k]    = rr
	L[k,k+1:] = (L[k,k+1:] + signW*s*x[k+1:])/c
	x[k+1:]   = c*x[k+1:]  - s*L[k, k+1:]

print "Covariance matrix"
print P
print "\nSquare root matrix with cholesky S"
print S
print "\nSquare root matrix with cholesky S*S'"
print np.dot(S,S.T)
print "\nSquare root matrix with QR + cholesky update L"
print L.T
print "\nSquare root matrix with QR + cholesky update L*L'"
print np.dot(L.T,L)