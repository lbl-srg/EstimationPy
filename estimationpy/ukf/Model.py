'''
===========================================
Module: **Model**
===========================================

The Module **Model** contains the class **Model** that is used as base class for representing the dynamic models used by
the Unscented Kalman Filter and the Smoother.

'''
import numpy as np

class Model(object):
    """
    
    This is an abstract class that represents a generic model, to be coupled with the UKF and Smoother.
    The model has to be represented in the following form
    
    .. math::
       \\frac{d x(t) }{dt} &=& F(x(t), u(t), \Theta, t)\\\\ 
       y(t_k) &=& G(x(t_k), u(t_k), \Theta, t_k)
    
    """
    
    def __init__(self, n_states, n_states_obs, n_pars, n_outputs, X0 = None, pars = None, r = None, q = None ):
        """
        Initialize the model with:
        
        * **n_states** is the number of state variables of the model,
        * **n_states_obs** is the number of state variables observed (to be estimated),
        * **n_pars** is the number of parameters to be identified,
        * **n_outputs** is the number of outputs of the system,
        * **X0** initial state values,
        * **pars** a tuple containing the parameters of the model,
        * **r** is an array containing the covariance of each output,
        * **q** is an array containing the covariance of each observed state variable,
        * 
         
        """
        
        # initialize state and output vectors
        self.n_states     = n_states
        self.n_states_obs = n_states_obs
        self.n_pars       = n_pars
        self.n_outputs    = n_outputs
        
        # initial state vector
        if X0 != None:
            self.X0 = X0
        else:
            self.X0 = np.zeros(self.n_statesTot)
            
        # set parameters of the model (Physical + control system)
        if pars != None:
            self.setPars(pars)

        # measurement covariance noise
        if r != None:
            self.R = np.diag(r)
        else:
            self.R = np.diag(np.zeros(self.n_outputs))
            
        self.sqrtR = np.linalg.cholesky(self.R)
         
        # process noise
        if q != None:
            self.Q     = np.diag(q)
        else:
            self.Q = np.diag(np.zeros(self.n_states))
        
        self.sqrtQ = np.linalg.cholesky(self.Q)
    
    
    def getNstates(self):
        """
        Get the total number of states variables
        """
        return self.n_states
    
    def getNstatesObs(self):
        """
        Get the number of observed states variables
        """
        return self.n_states

    
    def getNaugStates(self):
        """
        Get the number of all the augmented state variables
        """
        return self.n_states + self.n_pars

    
    def getNoutputs(self):
        """
        Get the number of output variables
        """
        return self.n_outputs
   
    
    def setInitialState(self,X0):
        """
        This method define the values of the initial state vector
        """
        self.X0 = X0    

    def getInitialState(self):
        """
        This method returns the value of the initial state vector
        """
        return self.X0

    def setPars(self, pars):
        """
        This method assigns the parameters of the model. They are contained in the tuple **pars**.
        """
        pass

    def setQ(self,Q):
        """
        This method define the observed state covariance matrix Q,
        and its square root computed with the Cholesky factorization.
        """
        self.Q     = Q
        self.sqrtQ = np.linalg.cholesky(Q)

    def setR(self,R):
        """
        This method define the output covariance matrix R,
        and its square root computed with the Cholesky factorization.
        """
        self.R     = R
        self.sqrtR = np.linalg.cholesky(R)        
           
    def getQ(self):
        """
        This method returns the observed state covariance matrix Q.
        """
        return self.Q

    def getR(self):
        """
        This method returns the output covariance matrix R.
        """
        return self.R
    
    def getSqrtQ(self):
        """
        This method returns the square root of the observed state covariance matrix Q.
        """
        return self.sqrtQ

    def getSqrtR(self):
        """
        This method returns the square root of the output covariance matrix R.
        """
        return self.sqrtR
    
    def functionF(self, val):
        """
        This function represents the dynamic behavior of the system described by the model. Given a tuple named *val*, structured as follows
        
        val = (x, u_old, u, t_old, t, other...)
        
        that contains
        
        * **x** is the value of the augmented state vector at time *t_old*,
        * **u_old** is the old vector of inputs at time *t_old*,
        * **u** is the actual vector of inputs at time *t*,
        * **t_old** is the time at which starting the simulation,
        * **t** is the time when stopping the simulation
        * **other** are other flags or parameters to specify when simulating the model
        
        This function returns the new value of the state vector **x** at time *t*.
        
        """
        pass

    def functionG(self, x, par, u, t, other):
        """
        This function represents the output function of the model, given
        
        * **x** is the value of the augmented state vector at time *t*,
        * **u** is the actual vector of inputs at time *t*,
        * **t** is the actual time,
        * **other** are other flags or parameters to specify when computing the outputs
        
        This function returns the output vector **y** at time *t*.
        
        """
        pass 