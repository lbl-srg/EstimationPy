import numpy as np
import pandas as pd
import multiprocessing

from estimationpy.fmu_utils.fmu_pool import FmuPool

class UkfFmu():
    """
    This class represents an Unscented Kalman Filter (UKF) that can be used for the 
    state and parameter estimation of nonlinear dynamic systems represented
    by FMU models.
    
    This class uses an object ot type :class:`estimationpy.fmu_utils.model.Model` to 
    represent the system. The model once, instantiated and configured,
    already contains the data series associated to the measured inputs and outputs,
    states and parameters to estimates, covariances, and boundaries.
    See :mod:`estimationpy.fmu_utils.model` for more information.
    
    The class internally uses an :class:`estimationpy.fmu_utils.fmu_pool.FmuPool` to 
    run simulations in paraller over multiple processors. Please have a look to
    :mod:`estimationpy.fmu_utils.fmu_pool` for more information.

    """
    
    def __init__(self, model, n_proc = multiprocessing.cpu_count() - 1):
        """
        Constructor of the class that initializes an object that can be used to solve
        state and parameter estimation problems by using the UKF and smoothing algorithms.
        The constructor requires a model representing the systems which states and/or parameters
        have to be estimated.
        
        The method performs the following steps
        
        1. creates a reference to the models and instantiales an object of type \
        :class:`estimationpy.fmu_utils.fmu_pool.FmuPool` for running simulations in parallel,
        2. compute the number of sigma points to be used,
        3. define the parameters of the filter,
        4. compute the weights associated to each sigma point,
        5. initialize the constraints on the observed state variables,
        
        :param estimationpy.fmu_utils.model.Model model: the model which states and/or parameters
          have to be estimated.
        :param int n_proc: a positive integer that defines the number of processes that can be spawned
          when the simulations are run. Make sure this value is equal to 1 if the filtering or smoothing
          are executed as part of a Celery task. By default this value is equal to the number of 
          available processors minus one.
        
        :raises ValueError: The method raises an exception if the model associated to the filter
          does not have state or parameters to be estimated.
        :raises Exception: The method raises an exception if there are not measured outputs, the 
          number of states to estimate is higher than the total number of states, of the number of
          parameters to estimate is invalid.

        """
        
        # Set the model
        self.model = model
        
        # Instantiate the pool that will run the simulation in parallel
        self.pool = FmuPool(self.model, processes = n_proc, debug = False)
        
        # Set the number of states variables (total and observed), parameters estimated and outputs
        self.n_state = self.model.get_num_states()
        self.n_state_obs = self.model.get_num_variables()
        self.n_pars = self.model.get_num_parameters()
        self.n_outputs = self.model.get_num_measured_outputs()
        self.n_outputsTot= self.model.get_num_outputs()
        
        self.N = self.n_state_obs + self.n_pars

        if self.N <= 0:
            raise ValueError("The model must have at least one parameter or state to estimate")
        
        # some check
        if self.n_state_obs > self.n_state:
            msg = 'The number of observed states ('+str(self.n_state_obs)+') cannot be '
            msg+= 'higher that the number of states ('+str(self.n_state)+')!'
            raise Exception(msg)
        if self.n_pars < 0:
            raise Exception('The number of estimated parameters cannot be < 0')
        if self.n_outputs < 0:
            raise Exception('The number of outputs cannot be < 0')
        
        # compute the number of sigma points
        self.n_points = 1 + 2*self.N

        # define UKF parameters with default values
        self.set_ukf_params()
        
        # set the default constraints for the observed state variables (not active by default)
        self.constrStateHigh = self.model.get_constr_obs_states_high()
        self.constrStateLow = self.model.get_constr_obs_states_low()
        
        # Max and Min Value of the states constraints
        self.constrStateValueHigh = self.model.get_state_observed_max()
        self.constrStateValueLow  = self.model.get_state_observed_min()
        
        # set the default constraints for the estimated parameters (not active by default)
        self.constrParsHigh = self.model.get_constr_pars_high()
        self.constrParsLow = self.model.get_constr_pars_low()
        
        # Max and Min Value of the parameters constraints
        self.constrParsValueHigh = self.model.get_parameters_max()
        self.constrParsValueLow  = self.model.get_parameters_min()
    
    def __str__(self):
        """
        This method returns a string representation of the object with
        a brief description.
        
        :return: string representation of the object
        :rtype: string
        
        """
        string = "\nUKF algorithm for FMU model"
        string += "\nThe FMU model name is:                     "+self.model.get_fmu_name()
        string += "\nThe total number of state variables is:    "+str(self.n_state)
        string += "\nThe number of state variables observed is: "+str(self.n_state_obs)
        string += "\nThe number of parameters estimated is:     "+str(self.n_pars)
        string += "\nThe number of outputs used to estimate is: "+str(self.n_outputs)
        return string
        
    
    def set_default_ukf_params(self):
        """
        This method initializes the parameters of the UKF to their
        default values and then computes the weights by calling the method
        :func:`compute_weights`. The default values are
        
        +------------------+---------------------------------+
        | parameter name   |  value                          |
        +==================+=================================+
        | :math:`\\alpha`   |  0.01                           |
        +------------------+---------------------------------+
        | :math:`\\beta`    |   1                             |
        +------------------+---------------------------------+
        | :math:`k`        |   2                             |
        +------------------+---------------------------------+
        | :math:`\\lambda`  | :math:`2 \\alpha (N + k) - N`    |
        +------------------+---------------------------------+
        | :math:`\\sqrt{C}` | :math:`\\alpha \\sqrt{N+k}`       |
        +------------------+---------------------------------+
        
        where :math:`N` is the total number of states and parameter to estimate. 
                
        """
        self.alpha    = 0.01
        self.k        = 1
        self.beta     = 2
        
        n = self.N
        
        self.lambd    = (self.alpha**2)*(n + self.k) - n
        self.sqrtC    = self.alpha*np.sqrt(n + self.k)
        
        # compute the weights
        self.compute_weights()

    def set_ukf_params(self, alpha = 1.0/np.sqrt(3.0), beta = 2, k = None):
        """
        This method allows to set the parameters of the UKF.
                
        :param float alpha: The parameter :math:`\\alpha` of the UKF
        :param float beta: The parameter :math:`\\beta` of the UKF
        :param float k: The parameter :math:`k` of the UKF

        given these parameters, the method computes
        
        .. math::
            \\lambda  &= 2 \\alpha (N + k) - N \\\\
            \\sqrt{C} &= \\alpha \\sqrt{N+k}
        
        where :math:`N` is the total number of states and parameters to estimate.

        """
        self.alpha     = alpha
        self.beta      = beta
        
        n = self.N
        
        if k == None:
            self.k = 3 - n
        else:
            self.k = k
        
        self.lambd    = (self.alpha**2)*(n + self.k) - n
        self.sqrtC    = self.alpha*np.sqrt(self.k + n)
        
        # compute the weights
        self.compute_weights()

    def get_ukf_params(self):
        """
        This method returns a tuple containing the parameters of the UKF.
        The parameters in the return tuple are
        
        * :math:`\\alpha`,
        * :math:`\\beta`,
        * :math:`k`,
        * :math:`\\lambda`,
        * :math:`\\sqrt{C}, and
        * :math:`N`
        
        :return: tuple containing the parameters of the UKF
        :rtype: tuple
        """

        return (self.alpha, self.beta, self.k, self.lambd, self.sqrtC, self.N)
        
    def compute_weights(self):
        """
        This method computes the vector of weights used by the UKF filter.
        These weights are associated to each sigma point and are used to
        compute the mean value and the covariance of the estimation at each step
        of the fitering process.
                
        There are two types of weigth vectors
        
        * :math:`\\mathbf{W}_m` is used to compute the mean value
        * :math:`\\mathbf{W}_c` is used to compute the covariance

        .. math::
             \\mathbf{W}_m^{(0)} &=& \\lambda / (N + \\lambda) \\\\
             \\mathbf{W}_c^{(0)} &=& \\lambda / (N + \\lambda) + (1 - \\alpha^2 + \\beta) , \\\\
             \\mathbf{W}_m^{(i)} &=& 1 / 2(N + \\lambda) \\ , \\ i=1 \\dots 2N , \\\\
             \\mathbf{W}_c^{(i)} &=& 1 / 2(N + \\lambda) \\ , \\ i=1 \\dots 2N
        
        where :math:`N` is the length os the state vector. In our case it is equal to 
        total number of states and parameters estimated.
        Also, :math:`\\mathbf{N}_m^{(i)}` indicates the i-th element of the vector :math:`\\mathbf{W}_m`.

        """
        
        n = self.N
        
        self.W_m = np.zeros((1+2*n, 1))
        self.W_c = np.zeros((1+2*n, 1))
        
        self.W_m[0,0] = self.lambd/(n + self.lambd)
        self.W_c[0,0] = self.lambd/(n + self.lambd) + (1 - self.alpha**2 + self.beta)

        for i in range(2*n):
            self.W_m[i+1,0] = 1.0/(2.0*(n + self.lambd))
            self.W_c[i+1,0] = 1.0/(2.0*(n + self.lambd))

        return
    
    def get_weights(self):
        """
        This method returns a tuple that contains the vectors :math:`\\mathbf{W}_m` and :math:`\\mathbf{W}_c`
        containing the weights used by the UKF. Each vector is a **numpy.array**.
        
        :return: tuple with first element :math:`\\mathbf{W}_m`, and second :math:`\\mathbf{W}_c`
        :rtype: tuple
        """
        return (self.W_m, self.W_c)

    def square_root(self, A):
        """
        This method computes the square root of a square matrix :math:`A`.
        The method uses the Cholesky factorization provided by the linear algebra
        package in **numpy**. The matrix returned is a lower triangular 
        matrix.
        
        :param numpy.ndarray A: square matrix :math:`A`
        :return: square root of math:`A`, such that :math:`S S^T = A`. The
          matrix is lower triangular.

        :rtype: numpy.ndarray
                
        """
        sqrtA = np.linalg.cholesky(A)
        return sqrtA
    
    def constrained_state(self, x_A):
        """
        This method applies the constraints associated to the state variables and
        parameters being estimated to the state vector :math:`\\mathbf{x}^A`.
        The constraints are applied only to the states and parameters estimated.
        
        :param numpy.ndarray x_A: vector :math:`\mathbf{x}^A` containing the states to be constrained
        :return: the constrained version of :math:`\mathbf{x}^A`
        :rtype: numpy.ndarray
        
        :raises ValueError: the method raises an exception if the parameter vector has a shape
          that does not correspond to the total number of states and parameters to estimate.
        """

        if len(x_A) != self.n_state_obs + self.n_pars:
            raise ValueError("The vector provided as input is not correct, desired length is {0}, provided is {1}".format(self.N, len(x_A)))
        
        # Check for every observed state
        for i in range(self.n_state_obs):
        
            # if the constraint is active and the threshold is violated
            if self.constrStateHigh[i] and x_A[i] > self.constrStateValueHigh[i]:
                x_A[i] = self.constrStateValueHigh[i]
                
            # if the constraint is active and the threshold is violated    
            if self.constrStateLow[i] and x_A[i] < self.constrStateValueLow[i]:
                x_A[i] = self.constrStateValueLow[i]
                
        # Check for every observed state
        for i in range(self.n_pars):
        
            # if the constraint is active and the threshold is violated
            if self.constrParsHigh[i] and x_A[self.n_state_obs+i] > self.constrParsValueHigh[i]:
                x_A[self.n_state_obs+i] = self.constrParsValueHigh[i]
                
            # if the constraint is active and the threshold is violated    
            if self.constrParsLow[i] and x_A[self.n_state_obs+i] < self.constrParsValueLow[i]:
                x_A[self.n_state_obs+i] = self.constrParsValueLow[i]
        
        return x_A
                
    def compute_sigma_points(self, x, pars, sqrtP):
        """
        This method computes the sigma points, Its inputs are
        
        * :math:`\\mathbf{x}`  -- the state vector around the points will be propagated,
        * :math:`\\mathbf{x}^P`  -- the vector of parameters that are estimated,
        * :math:`\\sqrt{P}` -- the square root of the state covariance matrix :math:`P`,\
          this matrix is used to spread the sigma points before their propagation.
        
        The sigma points are computed as

        .. math::

            \\mathbf{x}^{(0)} &=& \\boldsymbol{\\mu} , \\\\
            \\mathbf{x}^{(i)} &=& \\boldsymbol{\\mu} + \\left [ \\sqrt{(n+\\lambda) P} \\right ]_i \\ , \\ i=1 \\dots n , \\\\
            \\mathbf{x}^{(i)} &=& \\boldsymbol{\\mu} - \\left [ \\sqrt{(n+\\lambda) P} \\right ]_{i-n} \\ , \\ i=n+1 \\dots 2n
        
        where :math:`\\mu` is the average of the vector :math:`\\mathbf{x}^A`, defined as
        
        .. math::
        
            \\mathbf{x}^A = \\left[ \\mathbf{x} \\ , \\ \\mathbf{x}^P \\right]
        
        and :math:`\\boldsymbol{\\mu}` is its average.
        
        :param numpy.array x: vector containing the estimated states,
        :param numpy.array pars: vector containing the estimated parameters,
        :param numpy.ndarray sqrtP: square root of the covariance matrix :math:`P`
        :return: a matrix that contains the sigma points, each row is a sigma point that is\
          a vector of state and parameters to be evaluated.
        :rtype: numpy.ndarray
        
        :raises ValueError: The method raises a value error if the input parameters\
          ``x`` or ``pars`` do not respect the dimensions of the observed states and estimated\
          parameters.

        """
        try:
            # reshape the state vector
            x = np.squeeze(x)
            x = x.reshape(1, self.n_state_obs)
        except ValueError:
            msg = "The vector of state variables has a wrong size"
            msg += "{0} instead of {1}".format(x.shape, self.n_state_obs)
            print msg
            raise ValueError(msg)
        
        try:
            # reshape the parameter vector
            pars = np.squeeze(pars)
            pars = pars.reshape(1, self.n_pars)
        except ValueError:
            msg = "The vector of parameters has a wrong size"
            msg += "{0} instead of {1}".format(pars.shape, self.n_pars)
            print msg
            raise ValueError(msg)
            
        # initialize the matrix of sigma points
        # the result is
        # [[0.0, 0.0, 0.0],
        #  [0.0, 0.0, 0.0],
        #  [0.0, 0.0, 0.0],
        #  [0.0, 0.0, 0.0],
        #      ....
        #  [0.0, 0.0, 0.0]]
        
        Xs = np.zeros((self.n_points, self.n_state_obs + self.n_pars))

        # Now using the sqrtP matrix that is lower triangular:
        # create the sigma points by adding and subtracting the rows of the matrix sqrtP, to the lines of Xs
        # [[s11, 0  , 0  ],
        #  [s12, s22, 0  ],
        #  [s13, s23, s33]]
        
        xs0 = np.hstack((x, pars))
            
        Xs[0,:] = xs0
        
        i = 1
        N = self.N
        for row in sqrtP:
            Xs[i,:]   = xs0
            Xs[i+N,:] = xs0
            
            nso = self.n_state_obs
            ns  = nso
            npa = self.n_pars
            
            try:
                
                Xs[i,  0:nso] += self.sqrtC*row[0:nso]
                Xs[i,  ns:ns+npa] += self.sqrtC*row[ns:]
                
                Xs[i+N,  0:nso] -= self.sqrtC*row[0:nso]
                Xs[i+N,  ns:] -= self.sqrtC*row[ns:]
                    
            except ValueError:
                msg = "Is not possible to generate the sigma points..."
                msg +="\nthe dimensions of the sqrtP matrix and the state and parameter vectors are not compatible"
                msg +="\n {0} and {1}".format(sqrtP.shape, Xs.shape)
                print msg
                raise ValueError(msg)
            
            # Introduce constraints on points
            Xs[i,:] = self.constrained_state(Xs[i,:])
            Xs[i+N,:] = self.constrained_state(Xs[i+N,:])
            
            i += 1
        
        return Xs

    def sigma_point_proj(self, x_A, t_old, t):
        """
        This method, given a set of sigma points represented by the vector :math:`\\mathbf{x}^A`,
        propagates them using the state transition function. The state transition function is 
        a simulation run from time :math:`t_{old}` to :math:`t`.
        The simulations are managed by a **FmuPool** object.
        
        :param numpy.ndarray x_A: the vector containing the sigma points to propagate
        :param datetime.datetime t_old: the start time for the simulation that computes the propagations
        :param datetime.datetime t: the final time for the simulation that computes the propagations

        :return: a tuple that contains 
          * the projected states (only the estimated ones + estimated parameters), \
          * the projected outputs (only the measured ones), \
          * the full projected states (both estimated and not), \
          * the full projected outputs (either measured or not).
        
        :rtype: tuple
        
        **Note:**
        If for any reason the results of the simulation pool is an empty dictionary,
        the method tries again to run the simulations up to the maximum number
        of simulations allowed ``MAX_RUN`. By default ``MAX_RUN`` is equal to 3.
                
        """
        row, col = np.shape(x_A)
        
        # initialize the vector of the NEW STATES
        X_proj = np.zeros((row, self.n_state_obs + self.n_pars))
        Z_proj = np.zeros((row, self.n_outputs))
        Xfull_proj = np.zeros((row, self.n_state))
        Zfull_proj = np.zeros((row, self.n_outputsTot))
        
        # from the sigma points, get the value of the states and parameters
        values = []
        for sigma in x_A:
            x = sigma[0:self.n_state_obs]
            pars = sigma[self.n_state_obs:self.n_state_obs+self.n_pars]
            temp = {"state":x, "parameters":pars}
            values.append(temp)

        # Run simulations in parallel, if the results are not provided run again until the
        # maximum number of run is reached
        MAX_RUN = 3
        runs = 0
        poolResults = self.pool.run(values, start = t_old, stop = t)
        while poolResults == {} and runs < MAX_RUN:
            poolResults = self.pool.run(values, start = t_old, stop = t)
        
        i = 0
        for r in poolResults:
            time, results = r[0]
            
            X  = results["__ALL_STATE__"]
            Xo = results["__OBS_STATE__"]
            p  = results["__PARAMS__"]
            o  = results["__OUTPUTS__"]
            o_all = results["__ALL_OUTPUTS__"]
            
            Xfull_proj[i,:] = X
            X_proj[i,0:self.n_state_obs] = Xo
            X_proj[i,self.n_state_obs:self.n_state_obs+self.n_pars] = p
            Z_proj[i,:] = o
            Zfull_proj[i,:] = o_all
            
            i += 1
            
        return X_proj, Z_proj, Xfull_proj, Zfull_proj

    def average_proj(self, x_proj):
        """
        This function averages the projection of the sigma points.
        The function can be used to compute the average of both the state vector or
        the measured outputs. The weigths vetcor used is :math:`\\mathbf{W}_m`.
        
        :param np.ndarray x_proj: the vector to average :math:`\\mathbf{x}`
        :return: the average of the vector computed as :math:`\\mathbf{W}_m^T \\mathbf{x}`
        :rtype: numpy.ndarray
        
        """
        # make sure that the shape is [1+2*n, ...]
        x_proj.reshape(self.n_points, -1)
        
        # dot product of the two matrices
        avg = np.dot(self.W_m.T, x_proj)
        
        return avg

    def compute_P(self, x, x_avg, Q):
        """
        This method computes the state covariance matrix :math:`P` as
        
        .. math::
            
            P^{(i,j)} = \\mathbf{W}_c^{(i)} (\\mathbf{x}^(i) - \\boldsymbol{\\mu})
            (\\mathbf{x}^(j) - \\boldsymbol{\\mu})+ Q^{(i,j)}
        
        where :math:`\\boldsymbol{\\mu}` is the average of the vector :math:`\\mathbf{x}`
        among the different sigma points.
        The method removes the not observed states from :math:`\\mathbf{x}` and computes
        the covariance matrix :math:`\\mathbf{P}`, that has size :math:`N_{o+p} \\times N_{o+p}`.

        :param numpy.array x: vector that conatins the full state of the system (estimated, not estimated states, as well
          the estimated parameters), this vector can be seen as the propagated sigma points
        :param numpy.array x_avg: vector that contains the average of the propagated sigma points
        :param numpy.ndarray Q: covariance matrix

        """
        # create a diagonal matrix containing the weights
        W = np.diag(self.W_c[:,0]).reshape(self.n_points, self.n_points)
        
        # subtract each sigma point with the average x_avg, and tale just the augmented state
        V = x - x_avg
        
        # compute the new covariance matrix
        Pnew = np.dot(np.dot(V.T, W), V) + Q
        return Pnew
        
    def compute_cov_z(self, z, z_avg, R):
        """
        This method computes the output covariance matrix :math:`C_Z`
        that is the covariance matrix of the outputs, corrected
        by the measurements covariance matrix :math:`R`.

        .. math::     

            C_{z_{i,j}} = W_{c_i} (\mathbf{z}_i - \mathbf{z}_{avg})^2 + R_{i,j}

        :param numpy.array z: vector containing the outputs
        :param numpy.array z_avg: vector containing the average of the outputs
        :param numpy.ndarray R: measurements covarnace matrix
        :returns: output covariance matrix :math:`C_z`
        :rtype: numpy.ndarray
                
        """
        W = np.diag(self.W_c[:,0]).reshape(self.n_points, self.n_points)

        V =  np.zeros(z.shape)
        for j in range(self.n_points):
            V[j,:]   = z[j,:] - z_avg[0]
        
        covZ = np.dot(np.dot(V.T,W),V) + R
        
        return covZ
    
    def compute_cov_x_z(self, x, x_avg, z, z_avg):
        """
        This method computes the cross covariance matrix :math:`C_{xz}`
        between the states and outputs vectors.
        
        .. math::     

            C_{xz_{i,j}} = W_{c_i} (\mathbf{x}_i - \mathbf{x}_{avg}) (\mathbf{z}_i - \mathbf{z}_{avg})

        :param numpy.array x: vector that conatins the full state of the system (estimated, not estimated states, as well
          the estimated parameters), this vector can be seen as the propagated sigma points
        :param numpy.array x_avg: vector that contains the average of the propagated sigma points
        :param numpy.array z: vector containing the outputs
        :param numpy.array z_avg: vector containing the average of the outputs
        
        :returns: state-outputs covariance matrix :math:`C_{xz}`
        :rtype: numpy.ndarray
                
        """
        W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)
            
        Vx = x - x_avg
        
        Vz = np.zeros(z.shape)
        for j in range(self.n_points):
            Vz[j,:]   = z[j,:] - z_avg[0]
    
        covXZ = np.dot(np.dot(Vx.T,W),Vz)
        
        return covXZ
    
    def compute_cov_x_x(self, x_new, x_new_avg, x, x_avg):
        """
        This method computes the state-state cross covariance matrix :math:`C_{xx}`.
        The different states are the state before and after the propagation.
        
        .. math::     

            C_{xx_{i,j}} = W_{c_i} (\mathbf{x}_i - \mathbf{x}_{avg}) (\mathbf{x}_i^{new} - \mathbf{x}_{avg}^{new})

        :param numpy.array x_new: vector that contains the full state of the system (estimated, not estimated states, as well
          the estimated parameters), this vector can be seen as the propagated sigma points
        :param numpy.array x_new_avg: vector that contains the average of the propagated sigma points
        :param numpy.array x: vector containing initial states before the progatation
        :param numpy.array x_avg: vector containing the average of the initial state before the propagation
        
        :returns: state-state covariance matrix :math:`C_{xx}`
        :rtype: numpy.ndarray
                
        """
        W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)
            
        Vx_new = x_new - x_new_avg
        Vx  = x - x_avg
    
        covXX = np.dot(np.dot(Vx.T,W),Vx_new)
        
        return covXX
    
    def compute_C_x_x(self, x_new, x):
        """
        This method computes the state-state cross covariance matrix :math:`C_{xx}`
        between the old state :math:`\\mathbf{x}` and the new state :math:`\\mathbf{x}_{new}`.

         .. math::     

            C_{xx_{i,j}} = W_{c_i} (\mathbf{x}_i - \mathbf{x}_{avg}) (\mathbf{x}_i^{new} - \mathbf{x}_{avg}^{new})

        **Note:** This is method is used by the smoothing process because during the smoothing
        the averages are not directly available and thus :func:`compute_cov_x_x` can't be
        used.
                
        :param numpy.array x_new: vector that contains the full state of the system (estimated, not estimated states, as well
          the estimated parameters), this vector can be seen as the propagated sigma points
        :param numpy.array x: vector containing initial states before the progatation
        
        :returns: state-state covariance matrix :math:`C_{xx}`
        :rtype: numpy.ndarray
                
        """
        W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)
        x_ave_next = self.average_proj(X_next)
        x_ave_now  = self.average_proj(X_now)
        
        Vnext = X_next - x_ave_next
        Vnow  = X_now - x_ave_now
    
        Cxx = np.dot(np.dot(Vnext.T, W), Vnow)
        return Cxx
    
    def compute_S(self, x_proj, x_ave, sqrt_Q, w = None):
        """
        This method computes the squared root covariance matrix using the QR decomposition
        combined with a Cholesky update.
        The matrix returned by this method is upper triangular.
        
        :param numpy.array x_proj: projected full state vector
        :param numpy.array x_avg: average of the full state vector
        :param numpy.ndarray sqrt_Q: square root process covariance matrix
        :param numpy.array w: vector that contains the weights to use during the
          update. If not specified the method uses the weights automatically computed
          by the filter.
        
        :return: the square root of the updated state covariance matrix. The matrix is 
          upper triangular.
        :rtype: nunmpy.ndarray

        """
        x_proj_obs = x_proj
        x_ave_obs = x_ave
        
        # Matrix of weights and signs of the weights
        if w == None:
            weights = np.sqrt(np.abs(self.W_c[:,0]))
            signs   = np.sign(self.W_c[:,0])
        else:
            weights = np.sqrt(np.abs(w))
            signs   = np.sign(w)
        
        # create matrix A that contains the error between the sigma points and the average
        A     = np.array([[]])
        i     = 0
        for x in x_proj_obs:
            error = signs[i]*weights[i]*(x - x_ave_obs)
            
            # ignore when i==0, this will be done in the update
            if i == 1:
                A = error.T
            elif i > 1:
                A = np.hstack((A, error.T))
            i += 1

        # Put on the side the matrix sqrt_Q, that have to be modified to fit the dimension of the augmenets state    
        A = np.hstack((A, sqrt_Q))
        
        # QR factorization
        q, L = np.linalg.qr(A.T)
        
        # Execute Cholesky update
        x = signs[0]*weights[0]*(x_proj_obs[0,] - x_ave_obs)
        L = self.chol_update(L, x.T, self.W_c[:,0])
        
        return L
        
    def compute_S_y(self, z_proj, z_ave, sqrt_R):
        """
        This method computes the squared root covariance matrix using the QR decomposition
        combined with a Cholesky update.
        
        :param numpy.array z_proj: projected full state vector
        :param numpy.array z_avg: average of the full state vector
        :param numpy.ndarray sqrt_R: square root process covariance matrix
        
        :return: the square root of the updated output covariance matrix
        :rtype: nunmpy.ndarray

        """
        # Matrix of weights and signs of the weights
        weights = np.sqrt( np.abs(self.W_c[:,0]) )
        signs   = np.sign( self.W_c[:,0] )
        
        # create matrix A that contains the error between the sigma points outputs and the average
        A     = np.array([[]])
        i     = 0
        for z in z_proj:
            error = signs[i]*weights[i]*(z - z_ave)
            if i == 1:
                A = error.T
            elif i > 1:
                A = np.hstack((A, error.T))
            i    += 1
            
        # put the square root R matrix on the side
        A = np.hstack((A, sqrt_R))
        
        # QR factorization
        q, L = np.linalg.qr(A.T)

        # Execute the Cholesky update
        z = signs[0]*weights[0]*(z_proj[0,] - z_ave)
        L = self.chol_update(L, z.T, self.W_c[:,0])
        
        return L
    
    def chol_update(self, L, X, W):
        """
        This method computes the Cholesky update of a matrix.
        
        :param numpy.ndarray L: lower triangular matrix computed with QR factorization
        :param numpy.array X: vector used to compute the covariance matrix. It can either be a vector
          representing the deviation of the state from its average, or the deviation from an output
          from its average.
        :param numpy.array W: vector of weights
        
        :return: the square root matrix computed using the Cholesky update
        :rtype: numpy.ndarray
          
        """
        # Copy the matrix
        Lc = L.copy()
        
        # Compute signs of the weights
        signs   = np.sign(W)
    
        # Start the Cholesky update and do it for every column
        # of matrix X
        
        (row, col) = X.shape
        
        for j in range(col):
            x = X[0:,j]
            
            for k in range(row):
                rr_arg    = Lc[k,k]**2 + signs[0]*x[k]**2
                rr        = 1e-8 if rr_arg < 0 else np.sqrt(rr_arg)
                c         = rr / Lc[k,k]
                s         = x[k] / Lc[k,k]
                Lc[k,k]    = rr
                Lc[k,k+1:] = (Lc[k,k+1:] + signs[0]*s*x[k+1:])/c
                x[k+1:]   = c*x[k+1:]  - s*Lc[k, k+1:]
        
        # Check for the presence of any NaN
        if np.any(np.isnan(Lc)):
            return L
        # Check for the presence of any +/- inf
        elif np.any(np.isinf(Lc)):
            return L
        else:
            return Lc
    
    def ukf_step(self, x, sqrtP, sqrtQ, sqrtR, t_old, t, z = None, verbose = False):
        """
        This method implements the basic step that constitutes the UKF algorithm.
        The main steps are two:
        
        1. predition of the new state by projection,
        2. correction of the projection using the measurements
        
        :param numpy.array x: initial state vector
        :param numpy.ndarray sqrtP: square root of the process covariance matrix
        :param numpy.ndarray sqrtQ: square root of the process covariance matrix
        :param numpy.ndarray sqrtR: square root of the measurements/outputs covariance matrix
        :param datetime.datetime t_old: initial time for running the simulaiton
        :param datetime.datetime t: final time for runnign the simulation
        :param numpy.array z: measured outputs at time ``t```. If not provided the method retieves
          the data automatically by calling the method :func:`estimationpy.fmu_utils.model.Model.get_measured_data_ouputs`.
        :param boolean verbose: this boolean flag defines the level of logging required, set the flag to True only
          in debug mode.

        :return: a tuple with the following variables
        
          * a vector containing the corrected state,
          * the corrected quare root of the state covariance matrix, 
          * the average of the measured outputs, 
          * the square root of the output covariance matrix,
          * the average of the complete output vector,
          * the average of the full corrected state vector    
        
        :rtype: tuple
        """
        if verbose:
            print "Start UKF step"
        
        # Get the parameters and the states to observe
        pars = x[self.n_state_obs:]
        x = x[:self.n_state_obs]
        
        # the list of sigma points (each sigma point can be an array, containing the state variables)
        # x, pars, sqrtP, sqrtQ = None, sqrtR = None
        Xs      = self.compute_sigma_points(x, pars, sqrtP)
        
        if verbose:
            print "Sigma point Xs"
            print Xs
    
        # compute the projected (state) points (each sigma points is propagated through the state transition function)
        X_proj, Z_proj, Xfull_proj, Zfull_proj = self.sigma_point_proj(Xs,t_old,t)
        
        if verbose:
            print "Projected sigma points"
            print X_proj
    
        # compute the average
        x_ave = self.average_proj(X_proj)
        Xfull_ave = self.average_proj(Xfull_proj)
        
        if verbose:
            print "Averaged projected sigma points"
            print x_ave
        
        if verbose:
            print "Averaged projected full state"
            print Xfull_ave
        
        # compute the new squared covariance matrix S
        Snew = self.compute_S(X_proj,x_ave,sqrtQ)
        
        if verbose:
            print "New squared S matrix"
            print Snew
        
        # redraw the sigma points, given the new covariance matrix
        x    = x_ave[0,0:self.n_state_obs]
        pars = x_ave[0,self.n_state_obs:]
        Xs   = self.compute_sigma_points(x, pars, Snew)
        
        # Merge the real full state and the new ones
        self.model.set_state(Xfull_ave[0])
        
        if verbose:
            print "New sigma points"
            print Xs

        # compute the projected (outputs) points (each sigma points is propagated through the output function, this should not require a simulation,
        # just the evaluation of a function since the output can be directly computed if the state vector and inputs are known )
        X_proj, Z_proj, Xfull_proj, Zfull_proj = self.sigma_point_proj(Xs,t,t)
        
        if verbose:
            print "Output projection of new sigma points"
            print Z_proj
            print "State re-projection"
            print X_proj

        # compute the average output
        Zave = self.average_proj(Z_proj)
        Zfull_ave = self.average_proj(Zfull_proj)
        
        if verbose:
            print "Averaged output projection of new sigma points"
            print "Yav =",Zave

        # compute the innovation covariance (relative to the output)
        Sy = self.compute_S_y(Z_proj,Zave,sqrtR)
        
        if verbose:
            print "Output squared covariance matrix"
            print "Sy =",Sy
        
        # compute the cross covariance matrix
        CovXZ = self.compute_cov_x_z(X_proj, x_ave, Z_proj, Zave)
        
        if verbose:
            print "State output covariance matrix"
            print "Cxy =",CovXZ
    
        # Data assimilation step
        # The information obtained in the prediction step are corrected with the information
        # obtained by the measurement of the outputs
        # In other terms, the Kalman Gain (for the correction) is computed
        firstDivision = np.linalg.lstsq(Sy.T,CovXZ.T)[0]
        K             = np.linalg.lstsq(Sy, firstDivision)[0]
        K             = K.T
        
        # Read the output value
        if z == None:
            z = self.model.get_measured_data_ouputs(t)
        
        if verbose:
            print "Measured Output data to be compared against simulations"
            print "Z=",z
            print "Z - Yav =",z.reshape(self.n_outputs,1)-Zave.T
            print "K=",K
        
        # State correction using the measurements
        X_corr = x_ave + np.dot(K,z.reshape(self.n_outputs,1)-Zave.T).T
        
        # If constraints are active, they are imposed in order to avoid the corrected value to fall outside
        X_corr[0,:] = self.constrained_state(X_corr[0,:])
        
        if verbose:
            print "New state corrected"
            print X_corr
        
        # The covariance matrix is corrected too
        U      = np.dot(K,Sy)
        
        if verbose:
            print "U is"
            print U
            print "Snew"
            print Snew
        
        S_corr = self.chol_update(Snew,U,-1*np.ones(self.n_state))
        
        if verbose:
            print "New Covariance matrix corrected"
            print S_corr
        
        # Apply the corrections to the model and then returns
        # Set observed states and parameters
        self.model.set_state_selected(X_corr[0,:self.n_state_obs])
        self.model.set_parameters_selected(X_corr[0,self.n_state_obs:])
        
        return (X_corr[0], S_corr, Zave, Sy, Zfull_ave, Xfull_ave[0])        
    
    def filter(self, start, stop, verbose = False, for_smoothing = False):
        """
        This method starts the filtering process. The filtering process
        is a loop of multiple calls of the basic method :func:`ukf_step`.
        
        :param datetime.datetime start: time stamp that indicates the beginning of the
          filtering period
        :param datetime.datetime stop: time stamp that identifies the end of the
          filtering period
        :param bool verbose: Boolean flag that indicates the level of verbosity required
          when logging. Set to True only in debug mode.
        :param bool for_smoothing: Boolean flag that indicates if the data computed by this method
          will be used by a smoother. If True, the function returns more data do the smoother
          can use it.
        
        :return: the method returns a tuple containinig
        
          * time vector containing the instants at which the filter corrected
            and estimated the states and/or parameters,
          * the estimated states and parameters,
          * the square root of the covariance matrix of the estimated states and parameters,
          * the measured outputs,
          * the square root of the covariance matrix of the outputs,
          * the full outputs of the model

          if ``for_smoothing==True``, the following variables are added
        
          * the full states of the model,
          * the square root of the process covariance matrix,
          * the square root of the measurements covariance matrix
        
          **Note:** please note that every vector and matrix returned by this method is a list that
          contains the vector/matrices for each time stamp of the filtering process.
        
        :rtype: tuple
        
        :raises Exception: The method raises an exception if there are problem during the filtering process,
          e.g., numerical problems regarding the estimation.
        """
        # Read the output measured data
        measuredOuts = self.model.get_measured_output_data_series()
        
        # Get the time vector 
        time = pd.to_datetime(measuredOuts[:,0], utc = True)
        
        # find the index of the closest matches for start and stop time
        ix_start, ix_stop = self.find_closest_matches(start, stop, time)
        
        # Initial conditions and other values
        x     = [np.hstack((self.model.get_state_observed_values(), self.model.get_parameter_values()))]
        x_full= [self.model.get_state()]
        sqrtP = [self.model.get_cov_matrix_state_pars()]
        sqrtQ = self.model.get_cov_matrix_state_pars()
        sqrtR = self.model.get_cov_matrix_outputs()
        y     = [measuredOuts[0,1:]]
        y_full= [measuredOuts[0,1:]]
        Sy    = [sqrtR]
        
        for i in range(ix_start+1, ix_stop):
            t_old = time[i-1]
            t = time[i]
            z = measuredOuts[i,1:]
            
            # Execute a filtering step
            try:
                X_corr, sP, Zave, S_y, Zfull_ave, X_full = self.ukf_step(x[i-1-ix_start], sqrtP[i-1-ix_start], sqrtQ, sqrtR, t_old, t, z, verbose=verbose)
            except Exception, e:
                print "Exception while running UKF step from {0} to {1}".format(t_old, t)
                print str(e)
                print "The state X is"
                print x[i-1-ix_start]
                print "The sqrtP matrix is"
                print sqrtP[i-1-ix_start]
                raise Exception("Problem while performing a UKF step")
                
            # Add data to the list    
            x.append(X_corr)
            sqrtP.append(sP)
            y.append(Zave)
            y_full.append(Zfull_ave)
            Sy.append(S_y)
            x_full.append(X_full)
        
        # The first of the overall output vector is missing, copy from the second element
        y_full[0] = y_full[1]
        
        if for_smoothing:
            return time[ix_start:ix_stop], x, sqrtP, y, Sy, y_full, x_full, sqrtQ, sqrtR
        else:
            return time[ix_start:ix_stop], x, sqrtP, y, Sy, y_full
    
    def filter_and_smooth(self, start, stop, verbose=False):
        """
        This method executes the filter and then the smoothing of the data
        """
        # Run the filter
        time, X, sqrtP, y, Sy, y_full, x_full, sqrtQ, sqrtR = self.filter(start, stop, verbose = False, for_smoothing = True)
        
        print "=============================="
        print "== Starting the Smoother  ===="
        print "=============================="
        
        # get the number of time steps        
        #s = np.reshape(time,(-1,1)).shape
        s = time.shape
        nTimeStep = s[0]
        
        # initialize the smoothed states and covariance matrix
        # the initial value of the smoothed state estimation are equal to the filtered ones
        Xsmooth = list(X)
        Ssmooth = list(sqrtP)
        Yfull_smooth = list(y_full)
        
        # iterating starting from the end and back
        # i : nTimeStep-2 -> 0
        #
        # From point i with an estimation x_ave[i], and S[i]
        # new sigma points are created and propagated, the result is a 
        # new vector of states X[i+1] (one for each sigma point)
        #
        # NOTE that at time i+1 there is available a corrected estimation of the state Xcorr[i+1]
        # thus the difference between these two states is back-propagated to the state at time i
        for i in range(nTimeStep-2,-1,-1):
            
            # reset the full state of the model
            self.model.set_state(x_full[i])
            
            # actual state estimation and covariance matrix
            x_i = Xsmooth[i]
            S_i = Ssmooth[i]
            
            # take the value of the state and parameters estimated
            x = x_i[:self.n_state_obs]
            pars = x_i[self.n_state_obs:]
            
            # define the sigma points
            Xs_i      = self.compute_sigma_points(x, pars, S_i)
            
            if verbose:
                    print "Sigma point Xs"
                    print Xs_i
            
            # mean of the sigma points
            Xs_i_ave    = x_i
            
            if verbose:
                print "Mean of the sigma points"
                print Xs_i_ave
                print "Simulate from",time[i],"to",time[i+1]
                
            # compute the projected (state) points (each sigma points is propagated through the state transition function)
            X_plus_1, Z_plus_1, Xfull_plus_1, Zfull_plus_1 = self.sigma_point_proj(Xs_i, time[i], time[i+1])
            
            if verbose:
                print "Propagated sigma points"
                print X_plus_1
            
            # average of the sigma points
            x_ave_plus_1 = self.average_proj(X_plus_1)
            
            if verbose:
                print "Averaged propagated sigma points"
                print x_ave_plus_1
            
            # compute the new covariance matrix
            Snew = self.compute_S(X_plus_1, x_ave_plus_1, sqrtQ)
            
            if verbose:
                print "Former S matrix used to draw the points"
                print S_i
                print "New Squared covaraince matrix"
                print Snew
            
            # compute the cross covariance matrix of the two states
            # (new state already corrected, coming from the "future", and the new just computed through the projection)
            Cxx  = self.compute_cov_x_x(X_plus_1, x_ave_plus_1, Xs_i, Xs_i_ave)
            
            if verbose:
                print "Cross state-state covariance matrix"
                print Cxx
            
            # gain for the back propagation
            firstDivision = np.linalg.lstsq(Snew.T, Cxx.T)[0]
            D             = np.linalg.lstsq(Snew, firstDivision)[0]
            #D             = D.T
            
            correction = np.dot(np.matrix(Xsmooth[i+1]) - x_ave_plus_1, D)
            if verbose:
                print "Old state"
                print X[i]
                print "Error:"
                print Xsmooth[i+1] - x_ave_plus_1
                print "Correction:"
                print correction
            
            # correction (i.e. smoothing, of the state estimation and covariance matrix)
            Xsmooth[i]  = X[i] + np.squeeze(np.array(correction[0,:]))
            
            # How to introduce constrained estimation
            Xsmooth[i]  = self.constrained_state(Xsmooth[i])
            
            X_proj, Z_proj, Xfull_proj, Zfull_proj = self.sigma_point_proj([Xsmooth[i]],time[i], time[i])
            Yfull_smooth[i] = Zfull_proj[0]
            
            V          = np.dot(D.T, Ssmooth[i+1] - Snew)
            Ssmooth[i] = self.chol_update(sqrtP[i], V, -1*np.ones(self.n_state_obs + self.n_pars))
            
            if verbose:
                print "New smoothed state"
                print Xsmooth[i]
                print "Ssmooth"
                print "difference",sqrtP[i]-Ssmooth[i]
                
                raw_input("?")
        
        # correct the shape of the last element that has not been smoothed
        Yfull_smooth[-1] = Yfull_smooth[-1][0]
        
        # Return the results of the filtering and smoothing
        return time, X, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth
            
    def smooth(self,time,Xhat,S,sqrtQ,U,m,verbose=False):
        """
        This methods contains all the steps that have to be performed by the UKF Smoother.
        """
        # initialize the smoothed states and covariance matrix
        # the initial value of the smoothed state estimation are equal to the filtered ones
        Xsmooth = Xhat.copy()
        Ssmooth = S.copy()
        
        # get the number of time steps        
        s = np.reshape(time,(-1,1)).shape
        nTimeStep = s[0]

        # iterating starting from the end and back
        # i : nTimeStep-2 -> 0
        #
        # From point i with an estimation x_ave[i], and S[i]
        # new sigma points are created and propagated, the result is a 
        # new vector of states X[i+1] (one for each sigma point)
        #
        # NOTE that at time i+1 there is available a corrected estimation of the state Xcorr[i+1]
        # thus the difference between these two states is back-propagated to the state at time i
        for i in range(nTimeStep-2,-1,-1):
            # actual state estimation and covariance matrix
            x_i = Xsmooth[i,:]
            S_i = Ssmooth[i,:,:]

            # compute the sigma points
            Xs_i        = self.compute_sigma_points(x_i, S_i)
            
            if verbose:
                print "Sigma points"
                print Xs_i
            
            # mean of the sigma points
            Xs_i_ave    = self.average_proj(Xs_i)
            
            if verbose:
                print "Mean of the sigma points"
                print Xs_i_ave
            
            # propagate the sigma points
            x_plus_1    = self.sigma_point_proj(m,Xs_i,U[i],U[i+1],time[i],time[i+1])
            
            if verbose:
                print "Propagated sigma points"
                print x_plus_1
            
            # average of the sigma points
            x_ave_plus_1 = self.average_proj(x_plus_1)
            
            if verbose:
                print "Averaged propagated sigma points"
                print x_ave_plus_1
            
            # compute the new covariance matrix
            Snew = self.compute_S(x_plus_1,x_ave_plus_1,sqrtQ)
            
            if verbose:
                print "New Squared covaraince matrix"
                print Snew
            
            # compute the cross covariance matrix of the two states
            # (new state already corrected, coming from the "future", and the new just computed through the projection)
            Cxx  = self.compute_C_x_x(x_plus_1,x_ave_plus_1,Xs_i,Xs_i_ave)
            
            if verbose:
                print "Cross state-state covariance matrix"
                print Cxx
            
            # gain for the back propagation
            firstDivision = np.linalg.lstsq(Snew.T, Cxx.T)[0]
            D             = np.linalg.lstsq(Snew, firstDivision)[0]
            D             = D.T
            
            if verbose:
                print "Old state"
                print Xhat[i,:]
                print "Error:"
                print Xsmooth[i+1,0:self.n_state_obs] - x_ave_plus_1[0,0:self.n_state_obs]
                print "Correction:"
                print np.dot(D, Xsmooth[i+1,0:self.n_state_obs] - x_ave_plus_1[0,0:self.n_state_obs])
                
            # correction (i.e. smoothing, of the state estimation and covariance matrix)
            Xsmooth[i,self.n_state_obs:]   = Xhat[i,self.n_state_obs:]
            Xsmooth[i,0:self.n_state_obs]  = Xhat[i,0:self.n_state_obs] + np.dot(D, Xsmooth[i+1,0:self.n_state_obs] - x_ave_plus_1[0,0:self.n_state_obs])
            
            # How to introduce constrained estimation
            Xsmooth[i,0:self.n_state_obs]  = self.constrained_state(Xsmooth[i,0:self.n_state_obs])
            
            if verbose:
                print "New smoothed state"
                print Xsmooth[i,:]
                raw_input("?")
            
            V              = np.dot(D,Ssmooth[i+1,:,:] - Snew)
            Ssmooth[i,:,:] = self.chol_update(S[i,:,:],V,-1*np.ones(self.n_state_obs))
            
        return (Xsmooth, Ssmooth)

    @staticmethod
    def find_closest_matches(start, stop, time):
        """
        Given the vector that contains all the time stamps over which the inputs and measured
        outputs are defined, the function identifies which of its elements in the parameter 
        vector ``time`` are the closest the parameters ``start`` and ``stop``.

        **Note:**
        The function assumes the parameter ``time`` is sorted.
                
        :param datetime.datetime start: the initial time stamp
        :param datetime.datetime start: the final time stamp
        :param list time: a datetime index for the data
        
        :return: a tuple that contains the selected start and stop elements from ``time``
          that are the closest to ``start`` and ``stop``.
        :rtype: tuple
                
        """
        import bisect
        
        # Check that start and stop times are within the acceptable time range
        if not (start >= time[0] and start <= time[-1]):
            raise IndexError("The start time has to be between the time range")
        
        if not (stop >= time[0] and stop <= time[-1]):
            raise IndexError("The stop time has to be between the time range")
        
        if not (stop >= start):
            raise ValueError("The stop time has to be after the start time")
        
        # Find the closest value 
        ix_start = bisect.bisect_left(time, start)
        ix_stop = bisect.bisect_right(time, stop)
        
        return (ix_start, ix_stop)
