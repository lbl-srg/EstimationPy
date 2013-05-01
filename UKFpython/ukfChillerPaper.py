import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure

import sys
sys.path.insert(0,'./models/chillerExamplePaper')
sys.path.insert(0,'./models/chillerExamplePaper/plotting')
sys.path.insert(0,'./ukf')

from model import *
from ukf import *
from ukfAugmented import *
from plotting import *

# time frame of the experiment
dt          = 0.5
DT          = 120.0
startTime   = 6.0*3600
stopTime    = 22.0*3600

numPoints   = int((stopTime - startTime)/dt)
time        = np.linspace(startTime, stopTime, numPoints)

ratio       = int(DT/dt)
numSamples  = int((stopTime - startTime)/DT)
timeSamples = np.linspace(startTime, stopTime, numSamples)

# define the model
m = model(Ti = 80.0, Td = 0.0, K  = -0.03, b=1.0, c=1.5, CSmax = 1.0, CSmin = 0.0, dt= dt, DT=DT)

# initialize state and output vectors
n_inputs  = 7
n_state   = 10
n_outputs = 4

X = np.zeros((numPoints,n_state))
Y = np.zeros((numPoints,n_outputs))

# inputs vector: [Tch_in, Tcw_in, Tsp, CMD_P1, CMD_P2, CMD_V1, CMD_V2]
U   = np.zeros((numPoints,n_inputs))

U[:,0] = np.interp(time,[8*3600,9*3600,10*3600,10.5*3600,12*3600,15*3600,18*3600,21*3600],[10,11,13,14,15,16,13,11])
U[:,1] = np.interp(time,[8*3600,9*3600,12*3600,15*3600,18*3600,21*3600],[16,18,19,18,17,16])
U[:,2] = np.interp(time,[8*3600,21*3600],[4,4])
U[:,3] = np.interp(time,[7*3600,7.5*3600,12*3600,12.1*3600,16*3600, 16.5*3600,21*3600,21.5*3600],[0.1,0.55,0.55,1,1,0.6,0.6,0.1])
U[:,4] = np.interp(time,[7*3600,7.5*3600,12*3600,12.1*3600,16*3600, 16.5*3600,21*3600,21.5*3600],[0.1,0.55,0.55,1,1,0.6,0.6,0.1])
U[:,5] = np.interp(time,[7*3600,7.5*3600,21*3600,21.5*3600],[0.1,1,1,0.1])
U[:,6] = np.interp(time,[7*3600,7.5*3600,21*3600,21.5*3600],[0.1,1,1,0.1])	

print "Input defined..."

# compute evolution of the system
for i in range(numPoints):
	if i==0:
		X[i,:] = m.getInitialState()
	else:	
		X[i,:]= m.functionF_dt(X[i-1,:],U[i-1,:],time[i-1],simulate=True)

	Y[i,:]   = m.functionG(X[i,:],U[i,:],time[i],simulate=True)

print "True system simulated..."
# THE TRUE SYSTEM END HERE
##################################################################

# output measurement covariance noise
R     = np.diag([1**2, 1**2, 2**2, 10000**2,])
sqrtR = np.linalg.cholesky(R)

# input measurements covariance noise
sqrtH = np.diag([0.5, 0.5, 0, 0, 0, 0, 0])

# Sampled and noisy outputs and inputs
Z     = np.zeros((numSamples,n_outputs))
Um    = np.zeros((numSamples,n_inputs))
Uukf  = np.zeros((numSamples,n_inputs))

for i in range(numSamples):
	# big error measurements for a while
	if time[i*ratio]>=18*3600 and time[i*ratio]<=18*3600+10*60 :
		Z[i]    = Y[i*ratio,:] + np.dot(sqrtR, np.random.uniform(-8.0, 8.0, (n_outputs,1))).T
		Um[i]   = U[i*ratio,:] + np.dot(sqrtH, np.random.uniform(-8.0, 8.0, (n_inputs,1))).T
	else:
		Z[i]  = Y[i*ratio,:] + np.dot(sqrtR, np.random.uniform(-1.0, 1.0, (n_outputs,1))).T
		Um[i] = U[i*ratio,:] + np.dot(sqrtH, np.random.uniform(-1.0, 1.0, (n_inputs,1))).T
		
	# the sampled input for the model, contains the control action of the controller 
	# instead of the set point
	Uukf[i] = Um[i]
	Uukf[i,2] = X[i*ratio,8]
	
print "Sampled input defined..."

"""
############################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE BASIC SIMULATION PROCEDURE

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xsim = np.zeros((numSamples,n_state))
Ysim = np.zeros((numSamples,n_outputs))

# initial knowledge
# that is wrong
X0_sim = np.array([25.0, 25.0, 4.0, 0, 0, 5.0])

# iteration of the simulation model
for i in range(numSamples):
	if i==0:	
		Xsim[i,:]   = X0_sim
		Ysim[i,:]   = m.functionG(Xsim[i,:],Um[i,:],timeSamples[i], simulate = False)
	else:
		Xsim[i,:] = m.functionF(Xsim[i-1,:],Um[i-1,:],Um[i,:],timeSamples[i-1],timeSamples[i],simulate = False)
		Ysim[i,:] = m.functionG(Xsim[i,:],Um[i,:],timeSamples[i],simulate = False)

print "Model of the system given noisy inputs simulated..."
# COMPUTE MSE
MSE_Tch = 0
MSE_Tcd = 0
for i in range(numSamples):
	MSE_Tch += (X[i,1] - Xsim[i,1])**2
	MSE_Tcd += (X[i,2] - Xsim[i,2])**2

print "MSE_sim[1]= "+str(MSE_Tch)
print "MSE_sim[2]= "+str(MSE_Tcd)	
"""

########################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE FILERING PROCEDURE
# define the model

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat = np.zeros((numSamples,7))
Yhat = np.zeros((numSamples,3))
S    = np.zeros((numSamples,5,5))
Sy   = np.zeros((numSamples,3,3))

# initial of the subset of the state
X0_hat = np.array([12.0, 18.0, 3.0, 0.2, -0.2, 200000, 200000 ])

Q0     = np.diag([0.5**2, 0.5**2, 0.3**2, 0.005**2, 0.005**2])
#Q0     = np.diag([1, 1, 0.5])
S0     = np.linalg.cholesky(Q0)

R0     = np.diag([1**2, 1**2, 5**2])
sqrtR0 = np.linalg.cholesky(R0)


# UKF parameters
UKFilter  = ukf(n_state=7, n_state_obs=5, n_outputs=3)

# Initialize parameters of the filter
UKFilter.setUKFparams()

# iteration of the UKF
for i in range(numSamples):
	if i==0:	
		Xhat[i,:]   = X0_hat
		S[i,:,:]    = S0
		Yhat[i,:]   = m.functionG(Xhat[i,:],Uukf[i,:],timeSamples[i],simulate = False)
		Sy[i,:,:]   = sqrtR0
	else:
		Xhat[i,:], S[i,:,:], Yhat[i,:], Sy[i,:,:] = UKFilter.ukf_step(Z[i,0:3],Xhat[i-1,:],S[i-1,:,:],S0,sqrtR0,Uukf[i-1,:],Uukf[i,:],timeSamples[i-1],timeSamples[i],m,verbose=False)

# converting the squared matrix into the covariance ones
P         = np.zeros(S.shape)
covY      = np.zeros(Sy.shape)
(N, I, J) = P.shape
for n in range(N):
	P[n,:,:]       = np.dot(S[n,:,:],S[n,:,:].T)
	covY[n,:,:]    = np.dot(Sy[n,:,:], Sy[n,:,:].T)

print "Model based state estimation of the system given noisy inputs simulated (UKF)..."

# COMPUTE MSE UKF
MSE_Tch = 0
MSE_Tcd = 0
for i in range(numSamples):
	MSE_Tch += (X[i,1] - Xhat[i,1])**2
	MSE_Tcd += (X[i,2] - Xhat[i,2])**2

print "MSE_ukf[1]= "+str(MSE_Tch)
print "MSE_ufk[2]= "+str(MSE_Tcd)



# smoothing the results
Xsmooth = np.zeros((numSamples,7))
Ssmooth = np.zeros((numSamples,5,5))
Xsmooth, Ssmooth = UKFilter.smooth(timeSamples,Xhat,S,S0,Uukf,m,verbose=False)

# converting the squared matrix into the covariance ones
Psmooth   = np.zeros(Ssmooth.shape)
(N, I, J) = P.shape
for n in range(N):
	Psmooth[n,:,:] = np.dot(Ssmooth[n,:,:],Ssmooth[n,:,:].T)

print "Model based state estimation of the system given noisy inputs simulated (UKF Smoother)..."

# COMPUTE MSE UKF
MSE_Tch = 0
MSE_Tcd = 0
for i in range(numSamples):
	MSE_Tch += (X[i,1] - Xsmooth[i,1])**2
	MSE_Tcd += (X[i,2] - Xsmooth[i,2])**2

print "MSE_smooth[1]= "+str(MSE_Tch)
print "MSE_smooth[2]= "+str(MSE_Tcd)

# PLOT
plotBasic(time,timeSamples,startTime,stopTime,X,Y,U,Um,Z,Xhat,P,Yhat,covY,Xsmooth,Psmooth)

"""
###############################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE AUGMENTED FILERING PROCEDURE

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat_Aug = np.zeros((numSamples,n_state))
Yhat_Aug = np.zeros((numSamples,n_outputs))
S_Aug    = np.zeros((numSamples,n_state,n_state))
Sy_Aug   = np.zeros((numSamples,n_outputs,n_outputs))

# initial knowledge
X0_hat_Aug = np.array([20.0, 25.0, 8.0, 0, 0, 4.0])
Q0         = Q
R0         = R
Sq0        = np.linalg.cholesky(Q0)
Sr0        = np.linalg.cholesky(R0)

# UKF parameters
UKFilter_Aug  = ukfAugmented(n_state,n_outputs)
UKFilter_Aug.setAugmentedPars(0.995, 1.0/np.sqrt(3.0), 0.1*np.diag(np.ones(n_state)))

# iteration of the UKF
for i in range(numSamples):
	
	if i==0:	
		Xhat_Aug[i,:] = X0_hat_Aug
		S_Aug[i,:,:]  = Sq0
		Yhat_Aug[i,:] = m.functionG(Xhat_Aug[i,:],U[i,:],time[i])
		Sy_Aug[i,:,:] = Sr0
		Sq            = Sq0
		alpha_s       = 0.005
	else:
		Xhat_Aug[i,:], S_Aug[i,:,:], Yhat_Aug[i,:], Sy_Aug[i,:,:], Sq, alpha_s = UKFilter_Aug.ukfAugmented_step(Z[i],Xhat_Aug[i-1,:],S_Aug[i-1,:,:],Sq,Sr0,alpha_s,Um[i-1,:],Um[i,:],timeSamples[i-1],timeSamples[i],m,False,True)

# converting the squared matrix into the covariance ones
P_Aug         = np.zeros(S.shape)
covY_Aug      = np.zeros(Sy.shape)
(N, I, J)     = P_Aug.shape
for n in range(N):
	P_Aug[n,:,:]       = np.dot(S_Aug[n,:,:],S_Aug[n,:,:].T)
	covY_Aug[n,:,:]    = np.dot(Sy_Aug[n,:,:], Sy_Aug[n,:,:].T)


plotBasic(time,timeSamples,startTime,stopTime,X,Y,U,Um,Z,Xsim,Ysim,Xhat_Aug,P_Aug,Yhat_Aug,covY_Aug)

# plot the results
plotResults(time,stopTime,X,Y,U,Um,Pch,Z,Xhat,Yhat,P,covY,Xsmooth,Psmooth,Xhat_Aug,P_Aug)
"""