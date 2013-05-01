import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure

import sys
sys.path.insert(0,'./models/chillerExampleCTRL')
sys.path.insert(0,'./models/chillerExampleCTRL/plotting')
sys.path.insert(0,'./ukf')

from model import *
from ukf import *
from ukfAugmented import *
from plotting import *

# time frame of the experiment
dt          = 2.0
DT          = 120.0
startTime   = 7.0*3600
stopTime    = 21.0*3600

numPoints   = int((stopTime - startTime)/dt)
time        = np.linspace(startTime, stopTime, numPoints)

ratio       = int(DT/dt)
numSamples  = int((stopTime - startTime)/DT)
timeSamples = np.linspace(startTime, stopTime, numSamples)

# the states of the system are (Tch, Tcw, COP, d, CS, Tch_sp)
# initial state vector is
X0 = np.array([18.0, 20.0, 5.5, 0, 0, 5.0])

# output covariance (Tch, Tcw, W)
R     = np.diag([1.0**2, 1.0**2, 500**2])
sqrtR = np.linalg.cholesky(R)

# input measurement noise (Tch_in, Tcw_in, Tsp)
H     = np.diag([1.0**2, 1.0**2, 0.01**2])
sqrtH = np.linalg.cholesky(H)

# define the model
m = model(Ti = 100.0, Td = 0.0, K  = -0.04, b=1.0, c=1.5, CSmax = 1.0, CSmin = 0.0, dt= dt, DT=DT)
m.setInitialState(X0)

# initialize state and output vectors
n_inputs  = 3
n_state   = m.getALLstates()
n_outputs = m.getNoutputs()

X = np.zeros((numPoints,n_state))
Y = np.zeros((numPoints,n_outputs))

# inputs vector: [Tch_in, Tcw_in, Tsp]
U   = np.zeros((numPoints,n_inputs))

# define the inputs
for i in range(numPoints):
	if time[i]>= .0 and time[i]<8*3600.0:
		U[i,0] += 12.0
		U[i,1] += 20.0
		U[i,2] += 5.0
	elif time[i]>= 8*3600.0 and time[i]<12*3600.0:
		U[i,0] += 13.5
		U[i,1] += 22.0
		U[i,2] += 5.0
	elif time[i]>= 12*3600.0 and time[i]<17*3600.0:
		U[i,0] += 14.5
		U[i,1] += 23.0
		U[i,2] += 3.5
	elif time[i]>= 17*3600.0:
		U[i,0] += 14
		U[i,1] += 20
		U[i,2] += 5		

print "Input defined..."

# compute evolution of the system
for i in range(numPoints):
	if i==0:
		X[i,:] = X0
	else:	
		X[i,:]= m.functionF_dt(X[i-1,:],U[i-1,:],time[i-1],simulate=True)

	Y[i,:]   = m.functionG(X[i,:],U[i,:],time[i],simulate=True)

print "True system simulated..."
# THE TRUE SYSTEM END HERE
##################################################################

# introduce som eperturbation on the model
m.setPars( Mch=60, Mcw=100, wch=4.5, wcw=8.5, Ti = 120.0, Td = 0.0, K  = -0.05, b=1.0, c=1.0)

# Sampled and noisy outputs and inputs
Z   = np.zeros((numSamples,n_outputs))
Um  = np.zeros((numSamples,n_inputs))

for i in range(numSamples):
	if time[i*ratio]>=18*3600 and time[i*ratio]<=18*3600+10*60 :
		Z[i]  = Y[i*ratio,:] + np.dot(sqrtR, np.random.uniform(-8.0, 8.0, (n_outputs,1))).T
		Um[i] = U[i*ratio,:] + np.dot(sqrtH, np.random.uniform(-8.0, 8.0, (n_inputs,1))).T
	else:
		Z[i]  = Y[i*ratio,:] + np.dot(sqrtR, np.random.uniform(-1.0, 1.0, (n_outputs,1))).T
		Um[i] = U[i*ratio,:] + np.dot(sqrtH, np.random.uniform(-1.0, 1.0, (n_inputs,1))).T

print "Sampled input defined..."
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

########################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE FILERING PROCEDURE
# define the model

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat = np.zeros((numSamples,n_state))
Yhat = np.zeros((numSamples,n_outputs))
S    = np.zeros((numSamples,3,3))
Sy   = np.zeros((numSamples,n_outputs,n_outputs))

# initial of the subset of the state
X0_hat = np.array([25.0, 25.0, 4.0, 0, 0, 5.0])
# Q0     = np.diag([1*2, 1*2, 0.3**2])
Q0     = np.diag([0.2*2, 0.2*2, 0.1**2])
S0     = np.linalg.cholesky(Q0)
# R0     = np.diag([0.5, 0.5, 100**2])
R0     = np.diag([0.5, 0.5, 5**2])
sqrtR0 = np.linalg.cholesky(R0)

# UKF parameters
UKFilter  = ukf(n_state=n_state, n_state_obs=3, n_outputs=n_outputs)

UKFilter.setDefaultUKFparams()
#UKFilter.setUKFparams(alpha = 0.04, beta = 2, k=0)

# iteration of the UKF
for i in range(numSamples):
	if i==0:	
		Xhat[i,:]   = X0_hat
		S[i,:,:]    = S0
		Yhat[i,:]   = m.functionG(Xhat[i,:],Um[i,:],timeSamples[i],simulate = False)
		Sy[i,:,:]   = sqrtR0
	else:
		Xhat[i,:], S[i,:,:], Yhat[i,:], Sy[i,:,:] = UKFilter.ukf_step(Z[i],Xhat[i-1,:],S[i-1,:,:],S0,sqrtR0,Um[i-1,:],Um[i,:],timeSamples[i-1],timeSamples[i],m,verbose=False)

# converting the squared matrix into the covariance ones
P         = np.zeros(S.shape)
covY      = np.zeros(Sy.shape)
(N, I, J) = P.shape
for n in range(N):
	P[n,:,:]       = np.dot(S[n,:,:],S[n,:,:].T)
	covY[n,:,:]    = np.dot(Sy[n,:,:], Sy[n,:,:].T)

print "Model based state estimation of the system given noisy inputs simulated (UKF)..."

# COMPUTE MSE
MSE_Tch = 0
MSE_Tcd = 0
for i in range(numSamples):
	MSE_Tch += (X[i,1] - Xhat[i,1])**2
	MSE_Tcd += (X[i,2] - Xhat[i,2])**2

print "MSE_ukf[1]= "+str(MSE_Tch)
print "MSE_ufk[2]= "+str(MSE_Tcd)



# smoothing the results
Xsmooth = np.zeros((numSamples,n_state))
Ssmooth = np.zeros((numSamples,n_state,n_state))
Xsmooth, Ssmooth = UKFilter.smooth(timeSamples,Xhat,S,S0,Um,m,verbose=False)

# converting the squared matrix into the covariance ones
Psmooth   = np.zeros(Ssmooth.shape)
(N, I, J) = P.shape
for n in range(N):
	Psmooth[n,:,:] = np.dot(Ssmooth[n,:,:],Ssmooth[n,:,:].T)

print "Model based state estimation of the system given noisy inputs simulated (UKF Smoother)..."

# PLOT
plotBasic(time,timeSamples,startTime,stopTime,X,Y,U,Um,Z,Xsim,Ysim,Xhat,P,Yhat,covY,Xsmooth,Psmooth)

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