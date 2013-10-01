import numpy as np
from time import time
from os import system

np.set_printoptions(precision=6)

system('clear')

# Number of points for computing the covariance matrix
n = 100
# True mean vector
#Xtrue = np.array([[2.0, 3.0, 15.4, 21.3, 0.5, -8.35, 8.88, 34.5, 56.6, 21.9, 9]])
Xtrue = np.random.uniform(-8.0, 27.5, (1, 300))

# Generate the sample for computing the covariance matrix
noUsed, N = Xtrue.shape
Xpoints = np.zeros((n,N))
for i in range(n):
	noise = np.random.uniform(-2.0,2.0,(1,N)) 
	Xpoints[i,:] = Xtrue + noise


# default covariance to be added
Q = 2.0*np.eye(N)

# definition of the weights
Weights = np.zeros(n)
for i in range(n):
	if i==0:
		Weights[i] = 0.5
	else:
		Weights[i] = (1.0 - Weights[0])/np.float(n-1)

#############################################################
## STANDARD METHOD WITH CHOLESKY DECOMPOSITION
i = 0
P = Q
timeCholesky = time()
for x in Xpoints:
	error = x - Xtrue 
	P     = P + Weights[i]*np.dot(error.T,error)
	i    += 1
S = np.linalg.cholesky(P)
timeCholesky = time() - timeCholesky

# error
eCh = P - np.dot(S,S.T)

#############################################################
## QR FACTORIZATION + CHOLESKY UPDATE
# doing the same with QR factorization + Cholesky Update
i     = 0
sqrtQ = np.linalg.cholesky(Q)
A     = np.array([[]])

timeQRCholUpdate = time()
for x in Xpoints:
	error = np.sqrt(Weights[i])*(x - Xtrue)
	if i==1:
		A = error.T
	elif i>1:
		A = np.hstack((A,error.T))
	i    += 1
A = np.hstack((A,sqrtQ))

q,r = np.linalg.qr(A.T,mode='full')

# NOW START THE CHOLESKY UPDATE
error0 = Xpoints[0,]-Xtrue

L = r.copy()
x = error0[0].copy()

absW  = np.abs(Weights[0])
signW = np.sign(Weights[0])
x     = signW*np.sqrt(absW)*x

for k in range(N):
	rr_arg    = L[k,k]**2 + signW*x[k]**2
	rr        = 0.0 if rr_arg < 0 else np.sqrt(rr_arg)
	c         = rr / L[k,k]
	s         = x[k] / L[k,k]
	L[k,k]    = rr
	L[k,k+1:] = (L[k,k+1:] + signW*s*x[k+1:])/c
	x[k+1:]   = c*x[k+1:]  - s*L[k, k+1:]
timeQRCholUpdate = time() - timeQRCholUpdate

# error
eQRCholUpdate = P - np.dot(L.T,L)

#############################################################
## OTHER METHOD WITH ITERATIVE CHOLESKY UPDATE
# start with this matrix and add iteratively the other contributions
L = sqrtQ

# doing the same with iterative Cholesky Update
i = 0
timeCholeskyUpdate = time()
for x in Xpoints:
	error = x - Xtrue

	absW  = np.abs(Weights[i])
	signW = np.sign(Weights[i])
	X     = signW*np.sqrt(absW)*error[0]
	
	for k in range(N):
		rr_arg    = L[k,k]**2 + signW*X[k]**2
		rr        = 0.0 if rr_arg < 0 else np.sqrt(rr_arg)
		c         = rr / L[k,k]
		s         = X[k] / L[k,k]
		L[k,k]    = rr
		L[k,k+1:] = (L[k,k+1:] + signW*s*X[k+1:])/c
		X[k+1:]   = c*X[k+1:]  - s*L[k, k+1:]
	
	i += 1
timeCholeskyUpdate = time() - timeCholeskyUpdate

# error
eChUpdate = P - np.dot(L.T,L)

print "Max error between square root matrix with cholesky S*S' = "+str(eCh.max())
print "Max error between square root matrix with iterative cholesky update L*L' = "+str(eChUpdate.max())
print "Max error between square root matrix with QR factorization and single cholesky update L*L' = "+str(eQRCholUpdate.max())
print "Time with standard Cholesky: "+str(timeCholesky*1000)
print "Time with iteratiuve Cholesky update: "+str(timeCholeskyUpdate*1000)
print "Time with QR + single Cholesky update: "+str(timeQRCholUpdate*1000)