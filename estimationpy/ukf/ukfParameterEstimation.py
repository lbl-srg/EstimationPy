'''
Created on Sep 19, 2013

@author: marco
'''
import numpy as np

from ukf import ukf

class ukfParameterEstimation(ukf):
    
    def __init__(self, n_state, n_state_obs, n_pars, n_outputs, augmented=False):
        ukf.__init__(self, n_state, n_state_obs, n_pars, n_outputs, augmented)
    
    def estimation_step(self, x, pars, u, z, t, m, sqrtPo, Q, R, verbose=False):        
        """
        Using the methods provided by the standard UKF, a parameter estimation procedure is defined
        
        * z measured output of the system,
        * x initial state of the system,
        * pars vector of the estimated parameters,
        * u vector of inputs,
        * t time vector,
        * m the model of the system,
        * Po initial covariance matrix of the state observed and parameters estimated,
        * Q state covariance matrix,
        * R output covariance matrix
        
        The procedure is the following:
        
        1. Run the filter forward and try to estimate the observed states and the parameters
        2. Run the smoother back to improve the estimation (Note that this step requires some information computed by the step 1)
        
        """
        # check the time series provided
        Nsample = len(t)
        if np.size(u, 0) != Nsample or np.size(z, 0)!= Nsample or Nsample < 1:
            print "ERROR:: The data provided are not consistent"
            return
        
        # Initialize the state of the model
        m.setInitialState(x)
        # Initialize the estimated parameters 
        m.setPars(pars)
        
        # define the process and output covariance matrices
        m.setQ(Q)
        m.setR(R)
        sqrtQ = m.getSqrtQ()
        sqrtR = m.getSqrtR()
        
        # square root of the augmented state covariance matrix (observed states + est. parameters) 
        sqrtP  = sqrtPo 
        
        # define the matrices that will contain the state estimation, and the
        # state covariance
        Xukf = np.zeros((Nsample, self.n_state_obs + self.n_pars))
        Sukf = np.zeros((Nsample, self.n_state_obs + self.n_pars, self.n_state_obs + self.n_pars))
        Yukf = np.zeros((Nsample, self.n_outputs))
        Xukf[0,:] = m.getAugState()
        Sukf[0,:] = sqrtPo
        
        # define the list that will contain the sigma points drawn and propagated at each time step
        Xs_starting = []
        Xs_arriving = []
        S_arriving  = []
        
        # Start the UKF
        for i in range(Nsample-1):
            
            # compute the sigma points
            Xs = self.computeSigmaPoints(m.getState(), m.getPars(), sqrtP, sqrtQ, sqrtR)
            if verbose:
                print "Particles"
                print Xs
            
            # compute the propagation of the sigma points
            Xs_new, Ys = self.sigmaPointProj(m, Xs, u[i], u[i+1], t[i], t[i+1])
            if verbose:
                print "Projected Particles"
                print Xs_new
                print "Outputs"
                print Ys
            
            # take the average of the new sigma points (propagated)
            Xs_new_avg = self.averageProj(Xs_new)
            if verbose:
                print "Average new state"
                print Xs_new_avg
            
            # compute the state covariance matrix
            S   = self.computeS(Xs_new, Xs_new_avg, sqrtQ)
            if verbose:
                print "old sqrtP"
                print sqrtP
                print "new S"
                print S
            
            # redraw sigma points
            Xs_redraw = self.computeSigmaPoints(Xs_new_avg[0,0:self.n_state], Xs_new_avg[0,self.n_state:self.n_state+self.n_pars], S, sqrtQ, sqrtR)
            if verbose:
                print "Projected re-drawn Particles"
                print Xs_redraw
            
            # re-compute the output
            temp, Ys = self.sigmaPointProj(m, Xs_redraw, u[i+1], u[i+1], t[i+1]-1e-6, t[i+1]) 
            if verbose:
                print "Outputs"
                print Ys
            
            # take the average of the output of the new sigma points (re drawn)
            Ys_avg = self.averageProj(Ys)
            if verbose:
                print "Average Outputs"
                print Ys_avg
            
            # compute the output and cross covariance matrix
            Syy = self.computeSy(Ys, Ys_avg, sqrtR)
            Cxy = self.computeCovXZ(Xs_new, Xs_new_avg, Ys, Ys_avg)
            if verbose:
                print "Sqrt covariance output matrix"
                print Syy
                print "Covariance state-output"
                print Cxy
            
            # compute the Kalman gain
            firstDivision = np.linalg.lstsq(Syy.T,Cxy.T)[0]
            K             = np.linalg.lstsq(Syy, firstDivision)[0]
            K             = K.T
            if verbose:    
                print "Kalman gain:"
                print K
            
            # correct the new state estimation
            xnew = self.__AugStateFromFullState__(Xs_new_avg) + np.dot(K, z[i+1,:].reshape(self.n_outputs,1) - Ys_avg.T).T
            #xnew = self.constrainedState(xnew)
            
            if verbose:
                print "Corrected state"
                print xnew
            
            # The covariance matrix is corrected too
            U     = np.dot(K,Syy)
            sqrtP = self.cholUpdate(S,U,-1*np.ones(self.n_state_obs+self.n_pars))
            
            # modify the current knowledge of the state and parameters in the model
            m.setAugState(xnew[0,:])
            
            # store the information about the estimation and the sigma points
            Xukf[i+1,:] = xnew[0,0:self.n_state+self.n_pars]
            Sukf[i+1,:,:] = sqrtP[0:self.n_state+self.n_pars, 0:self.n_state+self.n_pars]
            Yukf[i+1,:] = Ys_avg
            Xs_starting.append(Xs)
            Xs_arriving.append(Xs_new)
            S_arriving.append(S)
        
        # Initial output is not known
        Yukf[0,:] = Yukf[1,:]
        
        # Define the smoothed estimation and the smoothed covariance
        Xsmooth = Xukf.copy()
        Ssmooth = Sukf.copy()
        """
        # Start the smoother
        for i in range(Nsample-2, -1, -1):
            
            # modify the current knowledge of the state and parameters in the model
            m.setAugState(Xukf[i,:])
            
            # compute the sigma points
            Xs = self.computeSigmaPoints(Xukf[i,0:self.n_state], Xukf[i,self.n_state:self.n_state+self.n_pars], Sukf[i,:,:], sqrtQ, sqrtR)
            
            # compute the propagation of the signma points
            Xs_new, Ys = self.sigmaPointProj(m, Xs, u[i], u[i+1], t[i], t[i+1])
            
            # take the average of the new sigma points (propagated), and the outputs
            Xs_new_avg = self.averageProj(Xs_new)
            Ys_avg = self.averageProj(Ys)
            
            # compute the state covariance matrix
            S   = self.computeS(Xs_new, Xs_new_avg, sqrtQ)
            
            # compute cross covariance matrix between the states at time k and time k+1
            Cxx = self.computeCxx(Xs, Xs_new)
            
            # compute the correction action (as Kalman gain)       
            # for the back propagation
            firstDivision = np.linalg.lstsq(S.T, Cxx.T)[0]
            D             = np.linalg.lstsq(S, firstDivision)[0]
            D             = D.T
            
            # correct the state estimation providing the smoothing estimate
            Xsmooth[i,:] = Xukf[i,:] + np.dot(Xsmooth[i+1,:] - self.__AugStateFromFullState__(Xs_new_avg), D)
            
            # correct the covariance matrix of the process
            V              = np.dot(D, Ssmooth[i+1,:,:] - S)
            Ssmooth[i,:,:] = self.cholUpdate(Sukf[i,:,:], V, -1*np.ones(self.n_state_obs+self.n_pars))
        """
        
        return Xukf, Sukf, Yukf, Xsmooth, Ssmooth