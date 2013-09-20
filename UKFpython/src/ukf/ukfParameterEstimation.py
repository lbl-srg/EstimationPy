'''
Created on Sep 19, 2013

@author: marco
'''
import numpy as np

from ukf import ukf

class ukfParameterEstimation(ukf):
    
    def __init__(self, n_state, n_state_obs, n_outputs):
        ukf.__init__(self, n_state, n_state_obs, n_outputs)
    
    def estimation_step(self, z, x, u, t, m, So, sqrtQ, sqrtR, verbose=False):        
        """
        Using the methods provided by the standard UKF, a parameter estimation procedure is defined
        
        * z measured output of the system,
        * x initial state of the system,
        * u vector of inputs,
        * t time vector,
        * m the model of the system,
        * S square root covariance matrix of the state,
        * sqrtQ square root of the process covariance matrix,
        * sqrtR square root of the output covariance matrix
        
        """
        
        numPoints = len(t)
        
        # The UKF starts from the guessed initial value Xhat, and a guessed covariance matrix Q
        Xhat = np.zeros((numPoints,self.n_state))
        Yhat = np.zeros((numPoints,self.n_outputs))
        S    = np.zeros((numPoints,self.n_state,self.n_state))
        Sy   = np.zeros((numPoints,self.n_outputs,self.n_outputs))
        
        # iteration of the UKF
        for i in range(numPoints):
            
            if i==0:    
                Xhat[i,:]   = x
                S[i,:,:]    = So
                Yhat[i,:]   = m.functionG(x, u[i], t[i])
                Sy[i,:,:]   = m.sqrtR
            else:
                Xhat[i,:], S[i,:,:], Yhat[i,:], Sy[i,:,:] = self.ukf_step(z[i], Xhat[i-1,:], S[i-1,:,:], m.sqrtQ, m.sqrtR, u[i-1], u[i], t[i-1], t[i], m, False)
        
        
        # smoothing the results        
        Xsmooth, Ssmooth = self.smooth(t,Xhat,S,m.sqrtQ,u,m)
        
        return Xhat, S, Yhat, Sy, Xsmooth, Ssmooth