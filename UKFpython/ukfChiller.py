import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure

import sys
sys.path.insert(0,'./models/chillerExample')
sys.path.insert(0,'./models/chillerExample/plotting')
sys.path.insert(0,'./ukf')

from model import *
from ukf import *
from plotting import *

# time frame of the experiment
dt = 10.0
startTime = 0.0
stopTime  = 3600.0
numPoints = int((stopTime - startTime)/dt)
time = np.linspace(startTime, stopTime, numPoints)

# the states of the system are (Tch, Tcw, COP)
# initial state vector is
X0 = np.array([25.0, 25.0, 5.5])

# output covariance (Tch, Tcw)
R     = np.diag([0.5, 0.5])
# state covariance (Tch, Tcw, Cop)
Q     = np.diag([0.1**2, 0.1**2, 0.1*1.0**2])

# input measurement noise (Tch_in, Tcw_in, PowerCompressor)
H     = np.diag([1.0**2, 1.0**2, 2000**2.0])
sqrtH = np.linalg.cholesky(H)

# define the model
m = model()
m.setInitialState(X0)
m.setDT(dt)
m.setQ(Q)
m.setR(R)

# initialize state and output vectors
n_inputs  = 3
n_state   = m.getNstates()
n_outputs = m.getNoutputs()

X = np.zeros((numPoints,n_state))
Y = np.zeros((numPoints,n_outputs))
Z = np.zeros((numPoints,n_outputs))

# inputs vector: [Tch_in, Tcw_in, W]
U   = np.zeros((numPoints,3))
Pch = np.zeros((numPoints,1))
Um  = np.zeros((numPoints,3))
for i in range(numPoints):
	if time[i]>= 0.0 and time[i]<1300.0:
		U[i,0] += 12.0
		U[i,1] += 24.0
		U[i,2] += 20000.0
	elif time[i]>= 1300.0 and time[i]<1500.0:
		U[i,0] += 13.0
		U[i,1] += 27.0
		U[i,2] += 20000.0+30000.0*(time[i]-1300.0)/200.0
	elif time[i]>= 1500.0:
		U[i,0] += 13.0 + 3.0*np.sin(2.0*np.pi*1/3600*(time[i] - 1500.0))
		U[i,1] += 27.0 - 1.0*np.sin(2.0*np.pi*1/1800*(time[i] - 1500.0))
		U[i,2] += 50000.0

# evolution of the system
for i in range(numPoints):
	if i==0:
		X[i,:] = X0
	else:	
		X[i,:]= m.functionF(X[i-1,:],U[i-1,:],time[i-1])

	Y[i,:]   = m.functionG(X[i,:],U[i,:],time[i])
	Pch[i,:] = m.functionPch(X[i,:],U[i,:],time[i])[0]

# noisy measured values (outputs)
Z  = Y + np.dot(m.sqrtR, np.random.uniform(-1.0, 1.0, (n_outputs,numPoints))).T
# noisy input measurements
Um = U + np.dot(sqrtH, np.random.uniform(-1.0, 1.0, (n_inputs,numPoints))).T


########################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE FILERING PROCEDURE

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat = np.zeros((numPoints,n_state))
Yhat = np.zeros((numPoints,n_outputs))
S    = np.zeros((numPoints,n_state,n_state))
Sy = np.zeros((numPoints,n_outputs,n_outputs))

# initial knowledge
X0_hat = np.array([23.0, 21.0, 3])
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

# plot the results
plotResults(time,stopTime,X,Y,U,Um,Pch,Z,Xhat,Yhat,P,covY,Xsmooth,Psmooth)
