'''
Created on Sep 19, 2013

@author: marco
'''
import numpy as np
from multiprocessing import Pool

from ukf import ukf
from ukf import Function

class ukf_Augmented(ukf):
    
    # Number of augmented states
    n_augState = 0
    
    # parameters for the augmented version of the UKF
    alpha_q  = 0
    lambda_s = 0
    mu       = 0
    A        = 0
    minS     = 0
    
    """
    initialization of the augmented UKF and its parameters
    """
    def __init__(self, n_state, n_state_obs, n_outputs):
    
        # recall explicitly the parent method
        ukf.__init__(self,n_state, n_state_obs, n_outputs)
    
        self.n_augState = 2*n_state_obs + n_outputs
        self.n_outputs  = n_outputs

        # compute the sigma points
        self.n_points  = 1 + 2*self.n_augState

        # define UKF parameters
        self.setUKFparams()

        # compute the weights
        self.computeWeights()
        
        # set parameters for the Augmented UKF modified version
        self.setAugmentedPars(0.995, 1/np.sqrt(3.0), 0.1*np.diag(np.ones(self.n_state)))
        
    """
    This method set the default parameters of the filter
    """
    def setDefaultUKFparams(self):
        self.alpha    = 0.01
        self.k        = 0
        self.beta     = 2
        self.lambd    = (self.alpha**2)*(self.n_augState + self.k) - self.n_augState
        self.sqrtC    = self.alpha*np.sqrt(self.n_augState + self.k)

    """
    This method set the non default parameters of the filter
    """
    def setUKFparams(self, alpha = 1.0/np.sqrt(3.0), beta = 2, k=None):
        self.alpha     = alpha
        self.beta      = beta

        if k == None:
            self.k = 3 - self.n_augState
        else:
            self.k = k
        
        self.lambd    = (self.alpha**2)*(self.n_augState + self.k) - self.n_augState
        self.sqrtC    = self.alpha*np.sqrt(self.k + self.n_augState)
    
    """
    This method set the parameters for the augmented version of the filter
    """
    def setAugmentedPars(self, alpha=0.99, mu= 1.0/np.sqrt(3.0), minS=0.0):
        # decay of the initial covariance
        self.alpha_q = alpha
        
        # growth of the computed covariance
        self.lambda_s = alpha
        
        # final value for the weight of the computed covariance
        self.mu       = mu
        self.A        = (1 - self.lambda_s)*self.mu
        
        # minimum covariance matrix allowed
        self.minS     = minS
        
    """
    This method computes the weights (both for covariance and mean value computation) of the augmented UKF filter
    """
    def computeWeights(self):
        self.W_m       = np.zeros((1+self.n_augState*2,1))
        self.W_c       = np.zeros((1+self.n_augState*2,1))
        
        self.W_m[0,0]  = self.lambd/(self.n_augState + self.lambd)
        self.W_c[0,0]  = self.lambd/(self.n_augState + self.lambd) + (1 - self.alpha**2 + self.beta)

        for i in range(2*self.n_augState):
            self.W_m[i+1,0] = 1.0/(2.0*(self.n_augState + self.lambd))
            self.W_c[i+1,0] = 1.0/(2.0*(self.n_augState + self.lambd))
    
    """
    comnputes the matrix of sigma points, given the square root of the covariance matrix
    and the previous state vector
    """
    def computeSigmaPoints(self,x,sqrtP):
        
        # reshape the state vector
        x = x.reshape(1,self.n_state + self.n_state_obs + self.n_outputs)

        # initialize the matrix of sigma points
        # the result is
        # [[0.0, 0.0, 0.0],
        #  [0.0, 0.0, 0.0],
        #  [0.0, 0.0, 0.0],
        #  [0.0, 0.0, 0.0],
        #      ....
        #  [0.0, 0.0, 0.0]]
        Xs      = np.zeros((self.n_points,self.n_state + self.n_state_obs + self.n_outputs))

        # Now using the sqrtP matrix that is lower triangular, I create the sigma points
        # by adding and subtracting the rows of the matrix sqrtP, to the lines of Xs
        # [[s11, 0  , 0  ],
        #  [s12, s22, 0  ],
        #  [s13, s23, s33]]
        #
        # The structure of teh state vector is
        # [Xobserved , sigmaX , sigmaY , Xunobserved]
        i = 1
        Xs[0,:] = x
        for row in sqrtP:
            Xs[i,:]                    = x
            Xs[i+self.n_augState,:]    = x
            
            Xs[i,0:self.n_augState]                 = x[0,0:self.n_augState] + self.sqrtC*row
            Xs[i+self.n_augState,0:self.n_augState] = x[0,0:self.n_augState] - self.sqrtC*row
            
            i += 1
        
        return Xs

    """
    This function computes the state evolution of all the sigma points through the model
    """
    def sigmaPointProj(self,m,Xs,u_old,u,t_old,t):
        # initialize the vector of the NEW STATES
        X_proj = np.zeros((self.n_points,self.n_state))
        
        # execute each state projection in parallel
        pool = Pool()
        res = pool.map_async(Function, ((m,np.hstack((x[0:self.n_state_obs],x[self.n_augState:])),u_old,u,t_old,t,False,) for x in Xs))
        pool.close()
        pool.join()
        
        j = 0
        for X in res.get():
            processNoise                  = Xs[j,self.n_state_obs:2*self.n_state_obs]
            X_proj[j,:]               = X
            X_proj[j,0:self.n_state_obs] += processNoise
            j += 1
            
        return X_proj        

    """
    This function computes the output measurement of all the sigma points through the model
    """
    def sigmaPointOutProj(self,m,Xs,u,t):
        # initialize the vector of the outputs
        Z_proj = np.zeros((self.n_points,self.n_outputs))
        
        j = 0
        for x in Xs:
            outputNoise = x[2*self.n_state_obs:2*self.n_state_obs+self.n_outputs]
            X           = np.hstack((x[0:self.n_state_obs],x[self.n_augState:]))
            
            Z_proj[j,:] = m.functionG(X,u,t,False) + outputNoise
            
            j += 1
        return Z_proj

    """
    This function computes the average of the sigma point evolution
    """
    def averageProj(self,X_proj):
        # make sure that the shape is [1+2*n, n]
        X_proj.reshape(1+self.n_augState*2, self.n_state)
        
        # dot product of the two matrices
        avg = np.dot(self.W_m.T, X_proj)
        return avg
    
    """
    This function computes the average of the sigma point outputs
    """
    def averageOutProj(self,Z_proj):
        # make sure that the stape is [1+2*n, n]
        Z_proj.reshape(1+self.n_augState*2, self.n_outputs)
        
        # dot product of the two matrices
        avg = np.dot(self.W_m.T, Z_proj)
        return avg
        
    """
    function call for prediction + update of the UKF
    """
    def ukf_step(self,z,x,S,Sq,Sr,alpha_s,u_old,u,t_old,t,m,verbose=False,adaptive=False):        
        
        # augmented state variables
        x_Aug = np.hstack((x[0:self.n_state_obs], np.zeros(self.n_state_obs), np.zeros(self.n_outputs), x[self.n_state_obs:]))
        
        if verbose:
            print "Augmnented initial state"
            print x_Aug
        
        # iterative weight of the computed covariance Sx        
        alpha_s  = (alpha_s*self.lambda_s + self.A)
        
        if not adaptive:
            actualSq = Sq
        else:
            # actual value of the process covariance
            actualSq = Sq + alpha_s*S
            if np.linalg.norm(actualSq,2) <= np.linalg.norm(self.minS,2):
                actualSq = self.minS
            # back propagate the initial squared covariance, reduced by alpha_q
            Sq      = self.alpha_q*Sq
        
        # augmented square covariance matrix
        S_1   = np.hstack((S,                                                  np.zeros((self.n_state_obs, self.n_state_obs)),    np.zeros((self.n_state_obs, self.n_outputs))))
        S_2   = np.hstack((np.zeros((self.n_state_obs, self.n_state_obs)),  actualSq,                                          np.zeros((self.n_state_obs, self.n_outputs))))
        S_3   = np.hstack((np.zeros((self.n_outputs, self.n_state_obs)),     np.zeros((self.n_outputs, self.n_state_obs)),     Sr))
        S_Aug = np.vstack((S_1, S_2, S_3))
        
        # the list of sigma points (each signa point can be an vector containing the state variables)
        Xs      = self.computeSigmaPoints(x_Aug,S_Aug)
        if verbose:
            print "Sigma Points"
            print Xs
    
        # compute the projected (state) points (each sigma points is propagated through the state transition function)
        X_proj = self.sigmaPointProj(m,Xs,u_old,u,t_old,t)
        if verbose:
            print "Projection of sigma points"
            print X_proj
    
        # compute the average
        Xave = self.averageProj(X_proj)
        if verbose:
            print "averaged projection"
            print Xave
        
        # compute the new squared covariance matrix S
        Snew = self.computeS(X_proj,Xave,actualSq)
        if verbose:
            print "Square root matrix of x"
            print Snew

        # compute the projected (outputs) points (each sigma points is propagated through the state transition function)
        Z_proj = self.sigmaPointOutProj(m,Xs,u,t)
        if verbose:
            print "Output projections"
            print Z_proj

        # compute the average output
        Zave = self.averageOutProj(Z_proj)
        if verbose:
            print "Average output projection"
            print Zave
        
        # compute the innovation covariance (relative to the output)
        Sy = self.computeSy(Z_proj,Zave,Sr)
        if verbose:
            print "Square root matrix of y"
            print Sy
        
        # compute the cross covariance matrix
        CovXZ = self.computeCovXZ(X_proj, Xave, Z_proj, Zave)
        if verbose:
            print "Covariance matrix between x and y"
            print CovXZ
        
        # Data assimilation step
        # The information obtained in the prediction step are corrected with the information
        # obtained by the measurement of the outputs
        
        firstDivision = np.linalg.lstsq(Sy.T,CovXZ.T)[0]
        K             = np.linalg.lstsq(Sy, firstDivision)[0]
        K             = K.T
        
        X_corr = Xave
        X_corr[:,0:self.n_state_obs] = Xave[:,0:self.n_state_obs] + np.dot(K,z.reshape(self.n_outputs,1)-Zave.T).T
        
        # How to introduce constrained estimation
        X_corr[0,0:self.n_state_obs] = self.constrainedState(X_corr[0,0:self.n_state_obs])
        
        U      = np.dot(K,Sy)
        S_corr = self.cholUpdate(Snew,U,-1*np.ones(self.n_state))
        
        
        
        

        return (X_corr, S_corr.T, Zave, Sy.T, Sq, alpha_s)