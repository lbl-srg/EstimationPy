import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure
from   scipy.stats import norm

import sys
import os
import time as TIM

sys.path.insert(0,'./models/chillerExamplePaper')
sys.path.insert(0,'./models/chillerExamplePaper/plotting')
sys.path.insert(0,'./ukf')

from model import *
from ukf import *
from ukfAugmented import *
from plotting import *

os.system("clear")

# time frame of the experiment
dt          = 0.5
DT          = 2.0*60.0
SmoothDT    = DT*30.0

startTime   = 6.0*3600
stopTime    = 22.0*3600

print "\nSimulating the Chiller model between: "+str(startTime/3600.0)+" - "+str(stopTime/3600.0)+" [hour]"
print "Step for simulation dt = "+str(dt)+" [s]"
print "Sampling time step DT = "+str(DT)+" [s]"
print "Smoothing window SmoothDT = "+str(SmoothDT)+" [s]"

numPoints   = int((stopTime - startTime)/dt)
time        = np.linspace(startTime, stopTime, numPoints)

ratio       = int(DT/dt)
numSamples  = int((stopTime - startTime)/DT)
timeSamples = np.linspace(startTime, stopTime, numSamples)

# define the model
m = model(Ti = 80.0, Td = 0.0, K  = -0.05, b=1.0, c=1.5, CSmax = 1.0, CSmin = 0.0, dt= dt, DT=DT)
m.plotEtaPL()

# initialize state and output vectors
n_inputs  = 7
n_state   = 10
n_outputs = 5

X = np.zeros((numPoints,n_state))
Y = np.zeros((numPoints,n_outputs))

# inputs vector: [Tch_in, Tcw_in, Tsp, CMD_P1, CMD_P2, CMD_V1, CMD_V2]
U   = np.zeros((numPoints,n_inputs))

#U[:,0] = np.interp(time,[8*3600,9*3600,10*3600,10.5*3600,12*3600,15*3600,18*3600,21*3600],[10,11,13,14,15,16,13,11])
U[:,0] = np.interp(time,[8*3600,9*3600,10*3600,10.5*3600,12*3600,15*3600,18*3600,21*3600],[9,10,12,13,14,15,12,10])
#U[:,1] = np.interp(time,[8*3600,9*3600,12*3600,15*3600,18*3600,21*3600],[16,18,19,18,17,16])
U[:,1] = np.interp(time,[8*3600,9*3600,12*3600,15*3600,18*3600,21*3600],[24,26,27,26,25,24])
U[:,2] = np.interp(time,[8*3600,21*3600],[3.9,3.9])
#U[:,3] = np.interp(time,[6*3600,7.5*3600,12*3600,12.1*3600,16*3600, 16.5*3600,21*3600,21.5*3600],[0.1,0.55,0.55,1,1,0.6,0.6,0.1])
U[:,3] = np.interp(time,[6*3600,7.5*3600,12*3600,12.1*3600,16*3600, 16.5*3600,21*3600,21.5*3600],[0.6,0.75,0.75,1,1,0.85,0.85,0.7])
#U[:,4] = np.interp(time,[6*3600,7.5*3600,12*3600,12.1*3600,16*3600, 16.5*3600,21*3600,21.5*3600],[0.1,0.55,0.55,1,1,0.6,0.6,0.1])
U[:,4] = np.interp(time,[6*3600,7.5*3600,12*3600,12.1*3600,16*3600, 16.5*3600,21*3600,21.5*3600],[0.6,0.75,0.75,1,1,0.85,0.85,0.7])
#U[:,5] = np.interp(time,[6*3600,7.5*3600,21*3600,21.5*3600],[0.1,1,1,0.1])
#U[:,6] = np.interp(time,[6*3600,7.5*3600,21*3600,21.5*3600],[0.1,1,1,0.1])	
U[:,5] = np.interp(time,[6*3600,7.5*3600,21*3600,21.5*3600],[0.2,0.5,0.5,0.2])
U[:,6] = np.interp(time,[6*3600,7.5*3600,21*3600,21.5*3600],[0.5,1,1,0.5])	


print "\n** Input defined..."

print "\n** Simulating..."
# compute evolution of the system
for i in range(numPoints):
	perc = int(100*(i+1)/float(numPoints))
	stTime = TIM.clock()
	if i==0:
		X[i,:] = m.getInitialState()
	else:	
		X[i,:]= m.functionF_dt(X[i-1,:],U[i-1,:],time[i-1],simulate=True)

	Y[i,:]   = m.functionG(X[i,:],U[i,:],time[i],simulate=True)
	
	endTime = TIM.clock()
	elapsedTime = endTime - stTime
	str2print = str(perc)+"% -- "+str(elapsedTime)+" s"
	sys.stdout.write("\r%s" %str2print)
	
# THE TRUE SYSTEM END HERE
##################################################################

# output measurement covariance noise
R     = np.diag([1.0**2, 1.0**2, 3**2, 10000**2, 0.2**2])
sqrtR = np.linalg.cholesky(R)

# input measurements covariance noise
sqrtH = np.diag([1.0, 1.0, 0, 0, 0, 0, 0])

# Sampled and noisy outputs and inputs
Z     = np.zeros((numSamples,n_outputs))
Um    = np.zeros((numSamples,n_inputs))
Uukf  = np.zeros((numSamples,n_inputs))


for i in range(numSamples):
	# big error measurements for a while
	if time[i*ratio]>=18*3600 and time[i*ratio]<=18*3600+20*60 :
		#Z[i]    = Y[i*ratio,:] + np.dot(sqrtR, np.random.uniform(-4.0, 4.0, (n_outputs,1))).T
		#Um[i]   = U[i*ratio,:] + np.dot(sqrtH, np.random.uniform(-4.0, 4.0, (n_inputs,1))).T
		Z[i]    = Y[i*ratio,:] + np.dot(2*sqrtR, np.random.standard_normal((n_outputs,1))).T
		Um[i]   = U[i*ratio,:] + np.dot(2*sqrtH, np.random.standard_normal((n_inputs,1))).T
	else:
		#Z[i]  = Y[i*ratio,:] + np.dot(sqrtR, np.random.uniform(-1.0, 1.0, (n_outputs,1))).T
		#Um[i] = U[i*ratio,:] + np.dot(sqrtH, np.random.uniform(-1.0, 1.0, (n_inputs,1))).T
		Z[i]  = Y[i*ratio,:] + np.dot(sqrtR, np.random.standard_normal((n_outputs,1))).T
		Um[i] = U[i*ratio,:] + np.dot(sqrtH, np.random.standard_normal((n_inputs,1))).T
		
	# the sampled input for the model, contains the control action of the controller 
	# instead of the set point
	Uukf[i] = Um[i]
	Uukf[i,2] = X[i*ratio,8]
	
print "\n\n** Sampled input defined..."

# COMPUTE MSE OF THE MEASUREMENTS
MSE_Tch = 0
MSE_Tcd = 0
for i in range(numSamples):
	MSE_Tch += (1/float(numSamples))*(X[i*ratio,0] - Z[i,0])**2
	MSE_Tcd += (1/float(numSamples))*(X[i*ratio,1] - Z[i,1])**2
MSE_Tch = np.sqrt(MSE_Tch)
MSE_Tcd = np.sqrt(MSE_Tcd)

print "MSE_measurements[1]= "+str(MSE_Tch)
print "MSE_measurements[2]= "+str(MSE_Tcd)

########################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE FILERING PROCEDURE
# define the model

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat = np.zeros((numSamples,7))
Yhat = np.zeros((numSamples,3))
S    = np.zeros((numSamples,5,5))
Sy   = np.zeros((numSamples,3,3))

# initial of the subset of the state
X0_hat = np.array([12.0, 18.0, 0.8, 0.05, 0.05, 200000, 200000 ])

#Q0     = np.diag([0.5**2, 0.5**2, 0.05**2, 0.005**2, 0.005**2])
#Q0     = np.diag([1.0**2, 1.0**2, 0.08**2, 0.02**2, 0.02**2])
Q0     = np.diag([2.0**2, 2.0**2, 0.1**2, 0.02**2, 0.03**2])
S0     = np.linalg.cholesky(Q0)

#R0     = np.diag([1**2, 1**2, 5**2])
#R0     = np.diag([2.5**2, 2.5**2, 5.0**2])
R0     = np.diag([4**2, 4**2, 5.0**2])
sqrtR0 = np.linalg.cholesky(R0)

# Set the constraints on the state variables
ConstrHigh = np.array([False, False, True, True, True])
ConstrLow  = np.array([False, False, True, True, True])
		
# Max Value of the constraints
ConstrValueHigh = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
# Min Value of the constraints
ConstrValueLow = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# UKF parameters
UKFilter  = ukf(n_state=7, n_state_obs=5, n_outputs=3)

# Initialize parameters of the filter
UKFilter.setUKFparams()

# Associate constraints
UKFilter.setHighConstraints(ConstrHigh, ConstrValueHigh)
UKFilter.setLowConstraints(ConstrLow, ConstrValueLow)


# iteration of the UKF
print "\n** Filtering..."
for i in range(numSamples):
	perc = int(100*(i+1)/float(numSamples))
	stTime = TIM.clock()
	
	if i==0:	
		Xhat[i,:]   = X0_hat
		S[i,:,:]    = S0
		Yhat[i,:]   = m.functionG(Xhat[i,:],Uukf[i,:],timeSamples[i],simulate = False)
		Sy[i,:,:]   = sqrtR0
	else:
		Xhat[i,:], S[i,:,:], Yhat[i,:], Sy[i,:,:] = UKFilter.ukf_step(Z[i,0:3],Xhat[i-1,:],S[i-1,:,:],S0,sqrtR0,Uukf[i-1,:],Uukf[i,:],timeSamples[i-1],timeSamples[i],m,verbose=False)
	
	endTime = TIM.clock()
	elapsedTime = endTime - stTime
	str2print = str(perc)+"% -- "+str(elapsedTime)+" s"
	sys.stdout.write("\r%s" %str2print)
	
	

# converting the squared matrix into the covariance ones
P         = np.zeros(S.shape)
covY      = np.zeros(Sy.shape)
(N, I, J) = P.shape
for n in range(N):
	P[n,:,:]       = np.dot(S[n,:,:],S[n,:,:].T)
	covY[n,:,:]    = np.dot(Sy[n,:,:], Sy[n,:,:].T)

# COMPUTE MSE UKF
MSE_Tch = 0.0
MSE_Tcd = 0.0
MSE_eta = 0.0
MSE_o1  = 0.0
MSE_o2  = 0.0
for i in range(numSamples):
	MSE_Tch += (1/float(numSamples))*(X[i*ratio,0] - Xhat[i,0])**2
	MSE_Tcd += (1/float(numSamples))*(X[i*ratio,1] - Xhat[i,1])**2
	MSE_eta += (1/float(numSamples))*(X[i*ratio,2] - Xhat[i,2])**2
	MSE_o1  += (1/float(numSamples))*(X[i*ratio,3] - Xhat[i,3])**2
	MSE_o2  += (1/float(numSamples))*(X[i*ratio,4] - Xhat[i,4])**2
MSE_Tch = np.sqrt(MSE_Tch)
MSE_Tcd = np.sqrt(MSE_Tcd)
MSE_eta = np.sqrt(MSE_eta)
MSE_o1 = np.sqrt(MSE_o1)
MSE_o2 = np.sqrt(MSE_o2)

print ""	
print "MSE_ukf[Tch]= "+str(MSE_Tch)
print "MSE_ufk[Tcd]= "+str(MSE_Tcd)
print "MSE_ufk[eta]= "+str(MSE_eta)
print "MSE_ufk[o1]= "+str(MSE_o1)
print "MSE_ufk[o2]= "+str(MSE_o2)

# Compute the percentage of points that are outside the sigma and 2-sigma boundary for each state and parameter estimated
pointsTch = 0.0
pointsTcd = 0.0
pointsEta = 0.0
pointsO1  = 0.0
pointsO2  = 0.0
for i in range(numSamples):
	pointsTch += 0.0 if np.abs(X[i*ratio,0] - Xhat[i,0]) < np.sqrt(P[i,0,0]) else 1.0
	pointsTcd += 0.0 if np.abs(X[i*ratio,1] - Xhat[i,1]) < np.sqrt(P[i,1,1]) else 1.0
	pointsEta += 0.0 if np.abs(X[i*ratio,2] - Xhat[i,2]) < np.sqrt(P[i,2,2]) else 1.0
	pointsO1  += 0.0 if np.abs(X[i*ratio,3] - Xhat[i,3]) < np.sqrt(P[i,3,3]) else 1.0
	pointsO2  += 0.0 if np.abs(X[i*ratio,4] - Xhat[i,4]) < np.sqrt(P[i,4,4]) else 1.0
	
print "out_sigma_ukf[Tch]= "+str(pointsTch/float(numSamples))+" [%]"
print "out_sigma_ukf[Tcd]= "+str(pointsTcd/float(numSamples))+" [%]"
print "out_sigma_ukf[eta]= "+str(pointsEta/float(numSamples))+" [%]"
print "out_sigma_ukf[o1]= "+str(pointsO1/float(numSamples))+" [%]"
print "out_sigma_ukf[o2]= "+str(pointsO2/float(numSamples))+" [%]"

# smoothing the results
Xsmooth = np.zeros((numSamples,7))
Ssmooth = np.zeros((numSamples,5,5))
print "\n** Smoothing..."

#OVERALL SMOOTHING
"""
Xsmooth, Ssmooth = UKFilter.smooth(timeSamples,Xhat,S,S0,Uukf,m,verbose=False)
"""
# NEW SMOOTHING PROCEDURE
# It should be done in parallel since it is quite slow...
NsmoothSteps = int(SmoothDT/DT)
Xsmooth = Xhat.copy()
Ssmooth = S.copy()
for i in range(numSamples-NsmoothSteps):
	perc = int(100*i/float(numSamples-NsmoothSteps))
	stTime = TIM.clock()
	
	Xsmooth[i:i+NsmoothSteps+1, :], Ssmooth[i:i+NsmoothSteps+1, :, :] = UKFilter.smooth(timeSamples[i:i+NsmoothSteps+1],Xhat[i:i+NsmoothSteps+1, :],S[i:i+NsmoothSteps+1, :, :],S0,Uukf[i:i+NsmoothSteps+1, :],m,verbose=False)
	
	endTime = TIM.clock()
	elapsedTime = endTime - stTime
	str2print = str(perc)+"% -- "+str(elapsedTime)+" s"
	sys.stdout.write("\r%s" %str2print)
	
# converting the squared matrix into the covariance ones
Psmooth   = np.zeros(Ssmooth.shape)
(N, I, J) = P.shape
for n in range(N):
	Psmooth[n,:,:] = np.dot(Ssmooth[n,:,:],Ssmooth[n,:,:].T)

# COMPUTE MSE UKF
MSE_Tch = 0.0
MSE_Tcd = 0.0
MSE_eta = 0.0
MSE_o1  = 0.0
MSE_o2  = 0.0
for i in range(numSamples):
	MSE_Tch += (1/float(numSamples))*(X[i*ratio,0] - Xsmooth[i,0])**2
	MSE_Tcd += (1/float(numSamples))*(X[i*ratio,1] - Xsmooth[i,1])**2
	MSE_eta += (1/float(numSamples))*(X[i*ratio,2] - Xsmooth[i,2])**2
	MSE_o1  += (1/float(numSamples))*(X[i*ratio,3] - Xsmooth[i,3])**2
	MSE_o2  += (1/float(numSamples))*(X[i*ratio,4] - Xsmooth[i,4])**2
MSE_Tch = np.sqrt(MSE_Tch)
MSE_Tcd = np.sqrt(MSE_Tcd)
MSE_eta = np.sqrt(MSE_eta)
MSE_o1 = np.sqrt(MSE_o1)
MSE_o2 = np.sqrt(MSE_o2)

print ""
print "MSE_smooth[Tch]= "+str(MSE_Tch)
print "MSE_smooth[Tcd]= "+str(MSE_Tcd)
print "MSE_smooth[eta]= "+str(MSE_eta)
print "MSE_smooth[o1]= "+str(MSE_o1)
print "MSE_smooth[o2]= "+str(MSE_o2)

# Compute the percentage of points that are outside the sigma and 2-sigma boundary for each state and parameter estimated
pointsTch = 0.0
pointsTcd = 0.0
pointsEta = 0.0
pointsO1  = 0.0
pointsO2  = 0.0
for i in range(numSamples):
	pointsTch += 0.0 if np.abs(X[i*ratio,0] - Xsmooth[i,0]) < np.sqrt(Psmooth[i,0,0]) else 1.0
	pointsTcd += 0.0 if np.abs(X[i*ratio,1] - Xsmooth[i,1]) < np.sqrt(Psmooth[i,1,1]) else 1.0
	pointsEta += 0.0 if np.abs(X[i*ratio,2] - Xsmooth[i,2]) < np.sqrt(Psmooth[i,2,2]) else 1.0
	pointsO1  += 0.0 if np.abs(X[i*ratio,3] - Xsmooth[i,3]) < np.sqrt(Psmooth[i,3,3]) else 1.0
	pointsO2  += 0.0 if np.abs(X[i*ratio,4] - Xsmooth[i,4]) < np.sqrt(Psmooth[i,4,4]) else 1.0
	
print "out_sigma_smooth[Tch]= "+str(pointsTch/float(numSamples))+" [%]"
print "out_sigma_smooth[Tcd]= "+str(pointsTcd/float(numSamples))+" [%]"
print "out_sigma_smooth[eta]= "+str(pointsEta/float(numSamples))+" [%]"
print "out_sigma_smooth[o1]= "+str(pointsO1/float(numSamples))+" [%]"
print "out_sigma_smooth[o2]= "+str(pointsO2/float(numSamples))+" [%]"

"""
###############################################################################################
# THE NOISY MEASUREMENTS OF THE OUTPUTS ARE AVAILABLE, START THE AUGMENTED FILERING PROCEDURE

# The UKF starts from the guessed initial value Xhat, and a guessed covarance matrix Q
Xhat_Aug = np.zeros((numSamples,7))
Yhat_Aug = np.zeros((numSamples,3))
S_Aug    = np.zeros((numSamples,5,5))
Sy_Aug   = np.zeros((numSamples,3,3))

# initial knowledge
X0_hat_Aug = np.array([12.0, 18.0, 0.8, 0.05, 0.05, 200000, 200000 ])

Q0     = np.diag([0.5**2, 0.5**2, 0.05**2, 0.01**2, 0.01**2])
R0     = np.diag([0.01**2, 0.01**2, 2**2])
Sq0        = np.linalg.cholesky(Q0)
Sr0        = np.linalg.cholesky(R0)

# UKF parameters
UKFilter_Aug  = ukf_Augmented(n_state=7, n_state_obs=5, n_outputs=3)
UKFilter_Aug.setAugmentedPars(alpha=0.9, mu=0.5, minS=0.1*np.diag(np.ones(5)))

# Associate constraints
UKFilter_Aug.setHighConstraints(ConstrHigh, ConstrValueHigh)
UKFilter_Aug.setLowConstraints(ConstrLow, ConstrValueLow)

# iteration of the UKF
for i in range(numSamples):
	
	if i==0:	
		Xhat_Aug[i,:] = X0_hat_Aug
		S_Aug[i,:,:]  = Sq0
		Yhat_Aug[i,:] = m.functionG(Xhat_Aug[i,:],Uukf[i,:],timeSamples[i],simulate = False)
		Sy_Aug[i,:,:] = Sr0
		Sq            = Sq0
		alpha_s       = 0.005
	else:
		Xhat_Aug[i,:], S_Aug[i,:,:], Yhat_Aug[i,:], Sy_Aug[i,:,:], Sq, alpha_s = UKFilter_Aug.ukf_step(Z[i,0:3],Xhat_Aug[i-1,:],S_Aug[i-1,:,:],Sq,Sr0,alpha_s,Uukf[i-1,:],Uukf[i,:],timeSamples[i-1],timeSamples[i],m,False,False)

# converting the squared matrix into the covariance ones
P_Aug         = np.zeros(S.shape)
covY_Aug      = np.zeros(Sy.shape)
(N, I, J)     = P_Aug.shape
for n in range(N):
	P_Aug[n,:,:]       = np.dot(S_Aug[n,:,:],S_Aug[n,:,:].T)
	covY_Aug[n,:,:]    = np.dot(Sy_Aug[n,:,:], Sy_Aug[n,:,:].T)
"""

# COMPUTE FAULT PROBABILITIES
# Define the thresholds
diffEta = 0.1
valveFault = 0.1

faultStatus = np.zeros((numSamples,2,4))
probFault   = np.zeros((numSamples,2,4))

for i in range(numSamples):
	faultStatus[i,0,0] = 0.0 if Xhat[i,2] - Y[i*ratio,4] < diffEta else 1.0
	faultStatus[i,1,0] = 0.0 if Xsmooth[i,2] - Y[i*ratio,4] < diffEta else 1.0
	
	faultStatus[i,0,1] = 0.0 if Xhat[i,2] - Y[i*ratio,4] > -diffEta else 1.0
	faultStatus[i,1,1] = 0.0 if Xsmooth[i,2] - Y[i*ratio,4] > -diffEta else 1.0
	
	faultStatus[i,0,2]  = 0.0 if Xhat[i,3] < valveFault else 1.0
	faultStatus[i,1,2]  = 0.0 if Xsmooth[i,3] < valveFault else 1.0
	
	faultStatus[i,0,3]  = 0.0 if Xhat[i,4] < valveFault else 1.0
	faultStatus[i,1,3]  = 0.0 if Xsmooth[i,4] < valveFault else 1.0
	
	# ComputingFault probabilities with smoothed esitation
	StdDev = np.diag(np.diag(Ssmooth[i,2:,2:]))
	D      = StdDev.copy()
	Dinv   = np.linalg.inv(D)
	R      = np.dot(Dinv, np.dot(Psmooth[i,2:,2:], Dinv))
	
	error = np.array([(Y[i*ratio,4]-diffEta) - Xsmooth[i,2], Xsmooth[i,3] - valveFault, Xsmooth[i,4] - valveFault])
	error = error/np.diag(StdDev)
	probFault[i,1,1:] = 100.0*norm.cdf(error)
	
	# ComputingFault probabilities with filtered esitation
	StdDev = np.diag(np.diag(S[i,2:,2:]))
	D      = StdDev.copy()
	Dinv   = np.linalg.inv(D)
	R      = np.dot(Dinv, np.dot(P[i,2:,2:], Dinv))
	
	error = np.array([(Y[i*ratio,4]-diffEta) - Xhat[i,2], Xhat[i,3] - valveFault, Xhat[i,4] - valveFault])
	error = error/np.diag(StdDev)
	probFault[i,0,1:] = 100.0*norm.cdf(error)

# PLOT
plotBasic(time,timeSamples,startTime,stopTime,X,Y,U,Um,Z,Xhat,P,Yhat,covY,Xsmooth,Psmooth,faultStatus,probFault)
# plotBasic(time,timeSamples,startTime,stopTime,X,Y,U,Um,Z,Xhat,P,Yhat,covY,Xhat_Aug,P_Aug)

