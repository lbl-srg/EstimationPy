'''
Created on Sep 19, 2013

@author: marco
'''
import numpy as np
import os

from models.firstOrderFMU.model import model
from models.firstOrderFMU.plotting import plotResults
from utilities.getCsvData import getCsvData

from ukf.ukf import ukf

os.system("clear")

# get the input and output from the csv file
dataMatrix = getCsvData("/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/data/NoisySimulationData_FirstOrder.csv")
simulationMatrix = getCsvData("/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/data/SimulationData_FirstOrder.csv")
(numPoints, columns) = np.shape(dataMatrix)

# time, input and output vectors (from CSV file)
time = dataMatrix[:,0]
U = dataMatrix[:,1]
X = dataMatrix[:,2] # This is the state to estimate, not known to the UKF
Z = dataMatrix[:,3]

# time, input and output vectors of the simulation (used to generate the data and unknown to the UKF)
timeSim = simulationMatrix[:,0]
Usim = simulationMatrix[:,1]
Ysim = simulationMatrix[:,3]

# parameters of the model
a = -1
b = 2.5
c = 3
d = 0.1
# initial state vector (position, velocity)
X0 = np.array([1.0])
# measurement covariance noise
R     = np.diag([1.0])
# process noise
Q     = np.diag([1.0])

# instantiate the model
m = model(a, b, c, d, X0)
m.setQ(Q)
m.setR(R)

# initialize state and output vectors
n_state   = m.getNstates()
n_outputs = m.getNoutputs()

# The UKF starts from the guessed initial value Xhat, and a guessed covariance matrix Q
Xhat = np.zeros((numPoints,n_state))
Yhat = np.zeros((numPoints,n_outputs))
S    = np.zeros((numPoints,n_state,n_state))
Sy   = np.zeros((numPoints,n_outputs,n_outputs))

# initial knowledge of the UKF
X0_hat = np.array([5.0])
S0     = m.sqrtQ

# UKF parameters
UKFilter  = ukf(n_state,n_state,n_outputs)

# iteration of the UKF
for i in range(numPoints):
    
    if i==0:    
        Xhat[i,:]   = X0_hat
        S[i,:,:]    = S0
        Yhat[i,:]   = m.functionG(X0_hat, U[i], time[i])
        Sy[i,:,:]   = m.sqrtR
    else:
        Xhat[i,:], S[i,:,:], Yhat[i,:], Sy[i,:,:] = UKFilter.ukf_step(Z[i], Xhat[i-1,:], S[i-1,:,:], m.sqrtQ, m.sqrtR, U[i-1], U[i], time[i-1], time[i], m, False)


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

###############################################################################################
# plot the results
plotResults(time, time[-1], U, X, Z, Xhat, Yhat, P, covY, Xsmooth, Psmooth, timeSim, Usim, Ysim)
