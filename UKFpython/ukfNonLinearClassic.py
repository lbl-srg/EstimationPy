import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure

import sys
sys.path.insert(0,'./models/nonLinearClassic')
sys.path.insert(0,'./models/nonLinearClassic/plotting')
sys.path.insert(0,'./ukf')

from model import *
from ukf import *
from plotting import *

# time frame of the experiment
dt = 1
startTime = 0.0
stopTime  = 100.0
numPoints = int((stopTime - startTime)/dt)
time = np.linspace(startTime, stopTime, numPoints)

# initial state vector (position, velocity)
X0 = np.array([10.0])

# measurement covariance noise
R  = np.array([[2.0]])

# process noise
Q  = np.array([[1.0]])

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
		X[i,:]= m.functionF(X[i-1,:],U[i-1,:],time[i-1]) + np.dot(m.sqrtR, np.random.randn(1,n_state))

	Y[i,:]  = m.functionG(X[i,:],U[i,:],time[i])

# noisy measured values
Z = Y + np.dot(m.sqrtR, np.random.randn(n_outputs,numPoints)).T

########################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE FILERING PROCEDURE

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat = np.zeros((numPoints,n_state))
Yhat = np.zeros((numPoints,n_outputs))
P    = np.zeros((numPoints,n_state,n_state))
CovZ = np.zeros((numPoints,n_outputs,n_outputs))

# initial knowledge
X0_hat = X0 + np.dot(m.sqrtQ,np.random.randn(n_state,1)).T
P0     = Q

# UKF parameters
UKFilter  = ukf(n_state,n_outputs)

# iteration of the UKF
for i in range(numPoints):
	
	if i==0:	
		Xhat[i,:]   = X0_hat
		P[i,:,:]    = P0
		Yhat[i,:]   = m.functionG(Xhat[i,:],U[i,:],time[i])
		CovZ[i,:,:] = m.R
	else:
		Xhat[i,:], P[i,:,:], Yhat[i,:], CovZ[i,:,:] = UKFilter.ukf_step(Z[i],Xhat[i-1,:],P[i-1,:,:],Q,R,U[i-1,:],U[i,:],time[i-1],time[i],m,False)

# smoothing the results
Xsmooth = np.zeros((numPoints,n_state))
Psmooth = np.zeros((numPoints,n_state,n_state))

Xsmooth, Psmooth = UKFilter.smooth(time,Xhat,P,Q,U,m)

# plot the results
plotResults(time,stopTime,X,Y,Z,Xhat,Yhat,P,CovZ,Xsmooth,Psmooth)

