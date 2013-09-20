import numpy as np

from models.stateParameterEstimation.model import model
from models.stateParameterEstimation.plotting import plotResults

from ukf.ukf import ukf
#from ukf.ukf import ukf_Augmented

# time frame of the experiment
dt = 0.1
startTime = 0.0
stopTime  = 30.0
numPoints = int((stopTime - startTime)/dt)
time = np.linspace(startTime, stopTime, numPoints)

# initial state vector (state x, par b)
X0 = np.array([3.5, 5.0])

# output measurement covariance noise
# R     = np.diag([0.1**2])
R     = np.diag([0.2**2])

# input measurement covariance noise
H     = np.diag([0.01*2])
sqrtH = np.linalg.cholesky(H)

# initial process noise
Q     = np.diag([0.05**2, 0.2**2])
#Q     = np.diag([1, 1])

# define the model
m = model()
m.setInitialState(X0)
m.setDT(dt)
m.setQ(Q)
m.setR(R)

# input vector
U = np.ones((numPoints,1))
# measured input vector
Um = np.ones((numPoints,1))

# initialize state and output vectors
n_state   = m.getNstates()
n_outputs = m.getNoutputs()

X = np.zeros((numPoints,n_state))
Y = np.zeros((numPoints,n_outputs))
Z = np.zeros((numPoints,n_outputs))

# evolution of the system
for i in range(numPoints):
	if i==0:
		X[i,:] = X0
	else:
		values = (X[i-1,:], U[i-1,:], U[i,:], time[i-1], time[i], True)
		X[i,:]= m.functionF(values)

	Y[i,:]  = m.functionG(X[i,:],U[i,:],time[i])
print "System simulated..."

# noisy measured values
Z  = Y + np.dot(m.sqrtR, np.random.randn(n_outputs,numPoints)).T
Um = U + np.dot(sqrtH, np.random.randn(1,numPoints)).T
print "Added noise..."

########################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE FILERING PROCEDURE

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat = np.zeros((numPoints,n_state))
Yhat = np.zeros((numPoints,n_outputs))
S    = np.zeros((numPoints,n_state,n_state))
Sy   = np.zeros((numPoints,n_outputs,n_outputs))

# initial knowledge
X0_hat = np.array([2.5, 2.0])
S0     = m.sqrtQ

# UKF parameters
UKFilter  = ukf(n_state, n_state, n_outputs)
#UKFilter.setUKFparams(0.1, 2, -1)

# iteration of the UKF
for i in range(numPoints):
	
	if i==0:	
		Xhat[i,:] = X0_hat
		S[i,:,:]  = S0
		Yhat[i,:] = m.functionG(Xhat[i,:],U[i,:],time[i])
		Sy[i,:,:] = m.sqrtR
	else:
		Xhat[i,:], S[i,:,:], Yhat[i,:], Sy[i,:,:] = UKFilter.ukf_step(Z[i],Xhat[i-1,:],S[i-1,:,:],m.sqrtQ,m.sqrtR,Um[i-1,:],Um[i,:],time[i-1],time[i],m,False)
print "UKF finished..."

# smoothing the results
Xsmooth = np.zeros((numPoints,n_state))
Ssmooth = np.zeros((numPoints,n_state,n_state))
Xsmooth, Ssmooth = UKFilter.smooth(time,Xhat,S,m.sqrtQ,Um,m)

# converting the squared matrix into the covariance ones
P         = np.zeros(S.shape)
Psmooth   = np.zeros(Ssmooth.shape)
covY      = np.zeros(Sy.shape)
(N, I, J) = P.shape
for n in range(N):
	P[n,:,:]       = np.dot(S[n,:,:],S[n,:,:].T) 
	Psmooth[n,:,:] = np.dot(Ssmooth[n,:,:],Ssmooth[n,:,:].T)
	covY[n,:,:]    = np.dot(Sy[n,:,:], Sy[n,:,:].T)
print "Smoother finished..."

###############################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE AUGMENTED FILERING PROCEDURE

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat_Aug = np.zeros((numPoints,n_state))
Yhat_Aug = np.zeros((numPoints,n_outputs))
S_Aug    = np.zeros((numPoints,n_state,n_state))
Sy_Aug   = np.zeros((numPoints,n_outputs,n_outputs))

# initial knowledge
X0_hat_Aug = np.array([2.5, 2.0])
Q0         = Q
R0         = R
Sq0        = np.linalg.cholesky(Q0)
Sr0        = np.linalg.cholesky(R0)

# UKF parameters
#UKFilter_Aug  = ukfAugmented(n_state,n_outputs)
#UKFilter_Aug.setAugmentedPars(0.995, 1.0/np.sqrt(3.0), 0.5*np.diag(np.ones(n_state)))

# iteration of the UKF
#for i in range(numPoints):
#	
#	if i==0:	
#		Xhat_Aug[i,:] = X0_hat_Aug
#		S_Aug[i,:,:]  = Sq0
#		Yhat_Aug[i,:] = m.functionG(Xhat_Aug[i,:],U[i,:],time[i])
#		Sy_Aug[i,:,:] = Sr0
#		Sq            = Sq0
#		alpha_s       = 0.005
#	else:
#		Xhat_Aug[i,:], S_Aug[i,:,:], Yhat_Aug[i,:], Sy_Aug[i,:,:], Sq, alpha_s = UKFilter_Aug.ukfAugmented_step(Z[i],Xhat_Aug[i-1,:],S_Aug[i-1,:,:],Sq,Sr0,alpha_s,Um[i-1,:],Um[i,:],time[i-1],time[i],m,False)
#
# converting the squared matrix into the covariance ones
P_Aug         = np.zeros(S.shape)
#covY_Aug      = np.zeros(Sy.shape)
#(N, I, J)     = P_Aug.shape
#for n in range(N):
#	P_Aug[n,:,:]       = np.dot(S_Aug[n,:,:],S_Aug[n,:,:].T)
#	covY_Aug[n,:,:]    = np.dot(Sy_Aug[n,:,:], Sy_Aug[n,:,:].T)

###############################################################################################
# plot the results
plotResults(time,stopTime,X,Y,Z,U,Um,Xhat,Yhat,P,covY,Xsmooth,Psmooth,Xhat_Aug,P_Aug)
