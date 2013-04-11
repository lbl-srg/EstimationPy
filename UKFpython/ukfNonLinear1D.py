import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure

import sys
sys.path.insert(0,'./models/nonLinear1D')
sys.path.insert(0,'./models/nonLinear1D/plotting')
sys.path.insert(0,'./ukf')

from model import *
from ukf import *
from plotting import *

# time frame of the experiment
dt = 0.01
startTime = 0.0
stopTime  = 1.0
numPoints = int((stopTime - startTime)/dt)
time = np.linspace(startTime, stopTime, numPoints)

# initial state vector (position, velocity)
X0 = np.array([1.0])

# measurement covariance noise
R     = np.diag([0.004])

# process noise
Q     = np.diag([0.01])

# define the model
m = model()
m.setInitialState(X0)
m.setDT(dt)
m.setQ(Q)
m.setR(R)

# input vector
U = np.zeros((numPoints,1))

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
		X[i,:]= m.functionF(X[i-1,:],U[i-1,:],time[i-1])

	Y[i,:]  = m.functionG(X[i,:],U[i,:],time[i])

# noisy measured values
Z = Y + np.dot(m.sqrtR, np.random.randn(n_outputs,numPoints)).T

########################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE FILERING PROCEDURE

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat = np.zeros((numPoints,n_state))
Yhat = np.zeros((numPoints,n_outputs))
S    = np.zeros((numPoints,n_state,n_state))
Sy   = np.zeros((numPoints,n_outputs,n_outputs))

# initial knowledge
X0_hat = X0 + np.dot(m.sqrtQ,np.random.randn(n_state,1)).T
S0     = m.sqrtQ

# UKF parameters
UKFilter  = ukf(n_state,n_outputs)

# iteration of the UKF
for i in range(numPoints):
	
	if i==0:	
		Xhat[i,:]   = X0_hat
		S[i,:,:]    = S0
		Yhat[i,:]   = m.functionG(Xhat[i,:],U[i,:],time[i])
		Sy[i,:,:]   = m.sqrtR
	else:
		Xhat[i,:], S[i,:,:], Yhat[i,:], Sy[i,:,:] = UKFilter.ukf_step(Z[i],Xhat[i-1,:],S[i-1,:,:],m.sqrtQ,m.sqrtR,U[i-1,:],U[i,:],time[i-1],time[i],m,False)


# smoothing the results
Xsmooth = np.zeros((numPoints,n_state))
Ssmooth = np.zeros((numPoints,n_state,n_state))
Xsmooth, Ssmooth = UKFilter.smooth(time,Xhat,S,m.sqrtQ,U,m)

# converting the squared matrix into the covariance ones
P         = np.zeros(S.shape)
Psmooth   = np.zeros(Ssmooth.shape)
covY      = np.zeros(Sy.shape)
(N, I, J) = P.shape
for n in range(N):
	P[n,:,:]       = np.dot(S[n,:,:],S[n,:,:].T) 
	Psmooth[n,:,:] = np.dot(Ssmooth[n,:,:],Ssmooth[n,:,:].T)
	covY[n,:,:]    = np.dot(Sy[n,:,:], Sy[n,:,:].T)

# plot the results
plotResults(time,stopTime,X,Y,Z,Xhat,Yhat,P,covY,Xsmooth,Psmooth)
