'''
Created on Sep 19, 2013

@author: marco
'''
import numpy as np
import os

from models.parameterIdentificationFMU.model import model
from models.parameterIdentificationFMU.plotting import plotResults, intermediatePlot
from utilities.getCsvData import getCsvData

from ukf.ukfParameterEstimation import ukfParameterEstimation

os.system("clear")

# get the input and output from the csv file
dataMatrix = getCsvData("/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/data/NoisySimulationData_FirstOrder.csv")
simulationMatrix = getCsvData("/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/data/SimulationData_FirstOrder.csv")
(numPoints, columns) = np.shape(dataMatrix)

# time, input and output vectors (from CSV file)
time = dataMatrix[:,0]
U = dataMatrix[:,1]
X = dataMatrix[:,2]
Z = dataMatrix[:,3]

# time, input and output vectors of the simulation (used to generate the data and unknown to the UKF)
timeSim = simulationMatrix[:,0]
Usim = simulationMatrix[:,1]
Ysim = simulationMatrix[:,3]

# unknown parameters of the model
x0 = 1.5
a = -0.1
b = 0.0
c = 0.0
d = 0.0
# initial state vector (position, velocity)
X0 = np.array([x0, a, b, c, d])
# measurement covariance noise
R     = np.diag([1.0])
# process noise
Q     = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])

# instantiate the model
m = model(X0)
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
X0_hat = X0
S0     = m.sqrtQ

# UKF for parameter estimation
estimator  = ukfParameterEstimation(n_state,n_state,n_outputs)
estimator.setUKFparams(0.01, 2, 1)

# Set the constraints on the state variables
ConstrHigh = np.array([True, True, False, False, False])
ConstrLow  = np.array([True, True, True, True, True])
        
# Max Value of the constraints
ConstrValueHigh = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

# Min Value of the constraints
ConstrValueLow = np.array([0.0, -10.0, 0.0, 0.0, 0.0])

# Associate constraints
estimator.setHighConstraints(ConstrHigh, ConstrValueHigh)
estimator.setLowConstraints(ConstrLow, ConstrValueLow)

# perform one step
pars = X0_hat[1:].copy()
i = 0
while i < 100:
    (Xhat, S, Yhat, Sy, Xsmooth, Ssmooth) = estimator.estimation_step(Z, X0_hat, U, time, m, S0, m.sqrtQ, m.sqrtR, False)
    X0_hat[0] = Xsmooth[0,0]
    X0_hat[1:] = Xsmooth[-1,1:]
    #X0_hat[1] = np.average(Xsmooth[:,1])
    #X0_hat[2] = np.average(Xsmooth[:,2])
    #X0_hat[3] = np.average(Xsmooth[:,3])
    #X0_hat[4] = np.average(Xsmooth[:,4])
    S0 = Ssmooth[0,:,:]
    pars = np.vstack((pars, np.average(Xsmooth[:,1:], 0)))
    i += 1
    print "Initial state: "+str(X0_hat[0])
    print "Parameters: \n"+str(pars)
    print "Covariances: "+str(np.diagonal(S0))
    #intermediatePlot(time, X, Z, Xhat, S, Yhat, Sy, Xsmooth, Ssmooth)

###############################################################################################
# plot the results
truePars = [-1.0, 2.5, 3.0, 0.1]
plotResults(pars, truePars)