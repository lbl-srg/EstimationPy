'''
Created on Sep 19, 2013

@author: marco
'''
import numpy as np

from models.parameterIdentificationFMU.model import model
from models.parameterIdentificationFMU.plotting import plotResults, intermediatePlot
from utilities.getCsvData import getCsvData

from ukf.ukfParameterEstimation import ukfParameterEstimation

# get the input and output from the csv file
dataMatrix = getCsvData("/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/data/NoisySimulationData_FirstOrder.csv")
simulationMatrix = getCsvData("/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/data/SimulationData_FirstOrder.csv")
(numPoints, columns) = np.shape(dataMatrix)

# time, input and output vectors (from CSV file)
time = dataMatrix[:,0]
U  = dataMatrix[:,1]
Y1 = dataMatrix[:,3]
Y2 = dataMatrix[:,2]
Y1 = np.matrix(Y1)
Y2 = np.matrix(Y2)
Z  = np.hstack((Y1.T, Y2.T))

# time, input and output vectors of the simulation (used to generate the data and unknown to the UKF)
timeSim = simulationMatrix[:,0]
Usim = simulationMatrix[:,1]
Xsim = simulationMatrix[:,2]
Ysim = simulationMatrix[:,3]

# Define the parameters of the model
x0 = 1.8
a = -0.5
b = 0.5
c = 0.5
d = 0.5
Q = np.diag([0.5])
R = np.diag([0.5, 0.5])
X0 = np.array([x0])
pars = np.array([a, b, c, d])

# instantiate the model
m = model(X0)
m.setPars(pars)
m.setQ(Q)
m.setR(R)

# instantiate the filter
filter = ukfParameterEstimation(1, 1, 4, 2, augmented = True)
# alpha, beta, k
filter.setUKFparams(0.05, 2.0, 1.0)
# Covariance matrix
Po = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
sqrtPo = np.linalg.cholesky(Po)
# define constraints for parameters
flags = [True, True, True, True]
valuesL = np.array([-10.0, 0.0, 0.0, 0.0])
valuesH = np.array([-0.1, 20.0, 20.0, 20.0])
filter.setParsLowConstraints(flags, valuesL)
filter.setParsHighConstraints(flags, valuesH)

# Initialize the parameters estimated
parV = pars

# perform the iterations
for i in range(400):
    Xhat, S, Yhat, Xsmooth, Ssmooth = filter.estimation_step(X0, pars, U, Z, time, m, sqrtPo, Q, R)
    X0     = np.array([Xsmooth[0,0]])
    pars   = Xsmooth[-1,1:]
    sqrtPo = S[-1,:,:]
    parV = np.vstack((parV, pars))
    print "Initial guess:"+str(Xhat[0,1:])
    print "Final guess:"+str(Xsmooth[0,1:])
    err1 = 0
    err2 = 0
    for i in range(len(Z)):
        err1 += (Z[i,0]-Yhat[i,0])*(Z[i,0]-Yhat[i,0])
        err2 += (Z[i,1]-Yhat[i,1])*(Z[i,1]-Yhat[i,1])
        
    print "Error: "+str(err1)+" -- "+str(err2)
    #intermediatePlot(time, timeSim, Xsim, Xhat, Yhat, Z, S, Xsmooth, Ssmooth)
    
intermediatePlot(time, timeSim, Xsim, Xhat, Yhat, Z, S, Xsmooth, Ssmooth)
plotResults(parV, [-1, 2.5, 3.0, 0.1])
    
