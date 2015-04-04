import numpy as np
import pandas as pd
import time as TIM

from estimationpy.fmu_utils.fmu_pool import FmuPool

class UkfFmu():
	"""
	This class represents an Unscented Kalman Filter (UKF) that can be used for the state and parameter estimation of nonlinear dynamic systems
	"""
	
	def __init__(self, model, augmented = False):
		"""
		Initialization of the UKF and its parameters. Provide the Model (FMU) as input
		
		The initialization assign these parameters then,
		
		1- compute the number of sigma points to be used
		2- define the parameters of the filter
		3- compute the weights associated to each sigma point
		4- initialize the constraints on the observed state variables
		 
		"""
		
		# Set the model
		self.model = model
		
		# Instantiate the pool that will run the simulation in parallel
		self.pool = FmuPool(self.model, debug = False)
		
		# Set the number of states variables (total and observed), parameters estimated and outputs
		self.n_state     = self.model.get_num_states()
		self.n_state_obs = self.model.get_num_variables()
		self.n_pars      = self.model.get_num_parameters()
		self.n_outputs   = self.model.get_num_measured_outputs()
		self.n_outputsTot= self.model.get_num_outputs()
		
		self.augmented = augmented
		if not augmented:
			self.N = self.n_state_obs + self.n_pars
		else:
			self.N = self.n_state_obs + self.n_pars + self.n_state_obs + self.n_pars + self.n_outputs
		
		# some check
		if self.n_state_obs > self.n_state:
			raise Exception('The number of observed states ('+str(self.n_state_obs)+') cannot be higher that the number of states ('+str(self.n_state)+')!')
		if self.n_pars < 0:
			raise Exception('The number of estimated parameters cannot be < 0')
		if self.n_outputs < 0:
			raise Exception('The number of outputs cannot be < 0')
		
		
		# compute the number of sigma points
		self.n_points    = 1 + 2*self.N

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
		This method returns a string that describe the object
		"""
		string  = "\nUKF algorithm for FMU model"
		string += "\nThe FMU model name is:                     "+self.model.get_fmu_name()
		string += "\nThe total number of state variables is:    "+str(self.n_state)
		string += "\nThe number of state variables observed is: "+str(self.n_state_obs)
		string += "\nThe number of parameters estimated is:     "+str(self.n_pars)
		string += "\nThe number of outputs used to estimate is: "+str(self.n_outputs)
		return string
		
	
	def set_default_ukf_params(self):
		"""
		This method set the default parameters of the UKF
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
		This method set the non default parameters of the UKF
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
	
	def compute_weights(self):
		"""
		This method computes the weights of the UKF filter. These weights are associated to each sigma point and are used to
		compute the mean value (W_m) and the covariance (W_c) of the estimation
		"""
		
		n = self.N
		
		self.W_m = np.zeros((1+2*n, 1))
		self.W_c = np.zeros((1+2*n, 1))
		
		self.W_m[0,0] = self.lambd/(n + self.lambd)
		self.W_c[0,0] = self.lambd/(n + self.lambd) + (1 - self.alpha**2 + self.beta)

		for i in range(2*n):
			self.W_m[i+1,0] = 1.0/(2.0*(n + self.lambd))
			self.W_c[i+1,0] = 1.0/(2.0*(n + self.lambd))
	
	def get_weights(self):
		"""
		This method returns the vectors containing the weights for the UKF
		"""
		return (self.W_m, self.W_c)

	def square_root(self, A):
		"""
		This method computes the square root of a square matrix A, using the Cholesky factorization
		"""
		try:
			sqrtA = np.linalg.cholesky(A)
			return sqrtA

		except np.linalg.linalg.LinAlgError:
			print "Matrix "+str(A)+" is not positive semi-definite"
			return A	
	
	def constrained_state(self, X):
		"""
		This method apply the constraints to the state vector (only to the estimated states)
		"""
			
		# Check for every observed state
		for i in range(self.n_state_obs):
		
			# if the constraint is active and the threshold is violated
			if self.constrStateHigh[i] and X[i] > self.constrStateValueHigh[i]:
				X[i] = self.constrStateValueHigh[i]
				
			# if the constraint is active and the threshold is violated	
			if self.constrStateLow[i] and X[i] < self.constrStateValueLow[i]:
				X[i] = self.constrStateValueLow[i]
				
		# Check for every observed state
		for i in range(self.n_pars):
		
			# if the constraint is active and the threshold is violated
			if self.constrParsHigh[i] and X[self.n_state_obs+i] > self.constrParsValueHigh[i]:
				X[self.n_state_obs+i] = self.constrParsValueHigh[i]
				
			# if the constraint is active and the threshold is violated	
			if self.constrParsLow[i] and X[self.n_state_obs+i] < self.constrParsValueLow[i]:
				X[self.n_state_obs+i] = self.constrParsValueLow[i]
		
		return X
				
	def compute_sigma_points(self, x, pars, sqrtP, sqrtQ = None, sqrtR = None):
		"""
		This method computes the sigma points, Its inputs are
		
		* x     -- the state vector around the points will be propagated,
		* pars  -- the parameters that are eventually estimated
		* sqrtP -- the square root matrix of the covariance P (both observed states and estimated parameters),
				   that is used to spread the sigma points
		
		"""
		try:
			# reshape the state vector
			x = np.squeeze(x)
			x = x.reshape(1, self.n_state_obs)
		except ValueError:
			print "The vector of state variables has a wrong size"
			print x
			print "It should be long: "+str(self.n_state_obs)
			return np.array([])
		
		try:
			# reshape the parameter vector
			pars = np.squeeze(pars)
			pars = pars.reshape(1, self.n_pars)
		except ValueError:
			print "The vector of parameters has a wrong size"
			print pars
			print "It should be long: "+str(self.n_pars)
			return np.array([])
			
		# initialize the matrix of sigma points
		# the result is
		# [[0.0, 0.0, 0.0],
		#  [0.0, 0.0, 0.0],
		#  [0.0, 0.0, 0.0],
		#  [0.0, 0.0, 0.0],
		#      ....
		#  [0.0, 0.0, 0.0]]
		
		if self.augmented:
			Xs = np.zeros((self.n_points, self.n_state_obs + self.n_pars + self.n_state_obs + self.n_pars + self.n_outputs))
		else:
			Xs = np.zeros((self.n_points, self.n_state_obs + self.n_pars))

		# Now using the sqrtP matrix that is lower triangular:
		# create the sigma points by adding and subtracting the rows of the matrix sqrtP, to the lines of Xs
		# [[s11, 0  , 0  ],
		#  [s12, s22, 0  ],
		#  [s13, s23, s33]]
		
		if self.augmented:
			zerosQ = np.zeros((1, self.n_state_obs + self.n_pars))
			zerosR = np.zeros((1, self.n_outputs))
			xs0    = np.hstack((x, pars, zerosQ, zerosR))
			
			zero1 = np.zeros((self.n_state_obs+self.n_pars, self.n_state_obs + self.n_pars))
			zero2 = np.zeros((self.n_state_obs+self.n_pars, self.n_outputs))
			zero3 = np.zeros((self.n_state_obs+self.n_pars, self.n_outputs))
			
			row1 = np.hstack((sqrtP,   zero1,   zero2)) 
			row2 = np.hstack((zero1.T, sqrtQ,   zero3))
			row3 = np.hstack((zero2.T, zero3.T, sqrtR))
			sqrtP = np.vstack((row1, row2, row3))
		else:
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
				
				if self.augmented:
					Xs[i,  0:nso] += self.sqrtC*row[0:nso]
					Xs[i,  ns:ns+npa] += self.sqrtC*row[ns:ns+npa]
					Xs[i,  ns+npa:ns+npa+nso] += self.sqrtC*row[ns+npa:ns+npa+nso]
					Xs[i,  ns+npa+nso:] += self.sqrtC*row[ns+npa+nso:]
					
					Xs[i+N,  0:nso] -= self.sqrtC*row[0:nso]
					Xs[i+N,  ns:ns+npa] -= self.sqrtC*row[ns:ns+npa]
					Xs[i+N,  ns+npa:ns+npa+nso] -= self.sqrtC*row[ns+npa:ns+npa+nso]
					Xs[i+N,  ns+npa+nso:] -= self.sqrtC*row[ns+npa+nso:]
				else:
					Xs[i,  0:nso] += self.sqrtC*row[0:nso]
					Xs[i,  ns:ns+npa] += self.sqrtC*row[ns:]
					
					Xs[i+N,  0:nso] -= self.sqrtC*row[0:nso]
					Xs[i+N,  ns:] -= self.sqrtC*row[ns:]
					
			except ValueError:
				print "Is not possible to generate the sigma points..."
				print "the dimensions of the sqrtP matrix and the state and parameter vectors are not compatible"
				return Xs
			
			# TODO:
			# How to introduce constrained points
			# Xs[i,0:self.n_state_obs] = self.constrained_state(Xs[i,0:self.n_state_obs])
			# Xs[i+self.n_state_obs,0:self.n_state_obs] = self.constrained_state(Xs[i+self.n_state_obs,0:self.n_state_obs])
			Xs[i,:] = self.constrained_state(Xs[i,:])
			Xs[i+N,:] = self.constrained_state(Xs[i+N,:])
			
			i += 1
		
		return Xs

	def sigma_point_proj(self, Xs, t_old, t):
		"""
		
		This function, given a set of sigma points Xs, propagate them using the state transition function.
		The simulations are run in parallel if the flag parallel is set to True
		
		"""
		row, col = np.shape(Xs)
		
		# initialize the vector of the NEW STATES
		X_proj = np.zeros((row, self.n_state_obs + self.n_pars))
		Z_proj = np.zeros((row, self.n_outputs))
		Xfull_proj = np.zeros((row, self.n_state))
		Zfull_proj = np.zeros((row, self.n_outputsTot))
		
		# from the sigma points, get the value of the states and parameters
		values = []
		for sigma in Xs:
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

	def average_proj(self,X_proj):
		"""
		This function averages the projection of the sigma points (both states and outputs)
		using a weighting vector W_m
		"""
		# make sure that the shape is [1+2*n, ...]
		X_proj.reshape(self.n_points, -1)
		
		# dot product of the two matrices
		avg = np.dot(self.W_m.T, X_proj)
		
		return avg

	def __aug_state_from_full_state__(self, Xfull):
		"""
		Given a vector that contains all the state variables of the models and the parameters to be identified,
		this method returns a vector that contains the augmented and observed states:
		[ observed states, parameters estimated]
		"""
		#return Xfull
		if False:
			row, col = np.shape(Xfull)
			Xaug = np.zeros((row, self.n_state_obs + self.n_pars + self.n_state_obs + self.n_outputs))
			
			nso = self.n_state_obs
			ns  = self.n_state 
			npa  = self.n_pars
			
			for i in range(row):
				Xaug[i, 0:nso] = Xfull[i, 0:nso] 
				Xaug[i, ns:ns+npa] = Xfull[i, ns:ns+npa]
				Xaug[i, ns+npa:ns+npa+nso] = Xfull[i, ns+npa:ns+npa+nso]
				Xaug[i, ns+npa+nso:] = Xfull[i, ns+npa+nso:]
				
		else:
			row, col = np.shape(Xfull)
			Xaug = np.zeros((row, self.n_state_obs + self.n_pars))
			
			for i in range(row):
				Xaug[i, 0:self.n_state_obs] = Xfull[i, 0:self.n_state_obs]
				Xaug[i, self.n_state_obs:] = Xfull[i, self.n_state_obs:self.n_state_obs+self.n_pars]
				
		return Xaug

	def __new_Q__(self, Q):
		"""
		This method, given the covariance matrix of the process noise (n_state_obs x n_state_obs)
		returns a new covariance matrix that has size (n_state_obs+n_pars x n_state_obs+n_pars)
		"""
		return Q
		nso = self.n_state_obs
		no  = self.n_outputs 
		npa  = self.n_pars
		if False:
			# create the new Q matrix to add
			A = np.zeros((nso, npa+nso+no))
			B = np.zeros((npa+nso+no, nso))
			C = np.zeros((npa+nso+no, npa+nso+no))
			top = np.hstack((Q, A))
			bot = np.hstack((B,C))
			newQ = np.vstack((top, bot))
		else:
			# create the new Q matrix to add
			A = np.zeros((nso, npa))
			B = np.zeros((npa, nso))
			C = np.zeros((npa, npa))
			top = np.hstack((Q, A))
			bot = np.hstack((B,C))
			newQ = np.vstack((top, bot))
		return newQ
		
	def compute_P(self, X_p, Xa, Q):
		"""
		This function computes the state covariance matrix P as
		
		P[i,j] = W_c[i]*(Xs[i] - Xavg)^2 + Q[i,j]
		
		The vectors X_ contain the all the states (observed and not) and the estimated parameters.
		The non observed states should be removed, and then computing P which has size of (n_state_obs + n_pars).
		Note that Q has size n_state_obs, thus it has to be expanded with zero elements when added.
		
		"""
		# create a diagonal matrix containing the weights
		W = np.diag(self.W_c[:,0]).reshape(self.n_points, self.n_points)
		
		# subtract each sigma point with the average Xa, and tale just the augmented state
		V = self.__aug_state_from_full_state__(X_p - Xa)
		
		# create the new Q matrix to add
		newQ = self.__new_Q__(Q)
		
		# compute the new covariance matrix
		Pnew = np.dot(np.dot(V.T, W), V) + newQ
		return Pnew
		
	def compute_cov_z(self, Z_p, Za, R):
		"""
		This function computes the output covariance matrix CovZ as
		
		CovZ[i,j] = W_c[i]*(Zs[i] - Zavg)^2 + R[i,j]
		
		"""
		W = np.diag(self.W_c[:,0]).reshape(self.n_points, self.n_points)

		V =  np.zeros(Z_p.shape)
		for j in range(self.n_points):
			V[j,:]   = Z_p[j,:] - Za[0]
		
		covZ = np.dot(np.dot(V.T,W),V) + R
		return covZ
	
	def compute_cov_x_z(self,X_p, Xa, Z_p, Za):
		"""
		This function computes the state-output cross covariance matrix (between X and Z)
		"""
		W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)
			
		Vx = self.__aug_state_from_full_state__(X_p - Xa)
		
		Vz = np.zeros(Z_p.shape)
		for j in range(self.n_points):
			Vz[j,:]   = Z_p[j,:] - Za[0]
	
		covXZ = np.dot(np.dot(Vx.T,W),Vz)
		return covXZ
	
	def compute_cov_x_x(self, X_p_1, Xa_1, X_p, Xa):
		"""
		This function computes the state-state cross covariance matrix (between X and Xnew)
		"""
		W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)
			
		Vx1 = self.__aug_state_from_full_state__(X_p_1 - Xa_1)
		Vx  = self.__aug_state_from_full_state__(X_p - Xa)
	
		covXX = np.dot(np.dot(Vx.T,W),Vx1)
		return covXX
	
	def compute_S(self, X_proj, Xave, sqrtQ):
		"""
		This function computes the squared covariance matrix using QR decomposition + a Cholesky update
		"""
		# take the augmented states of the sigma points vectors
		# that are the observed states + estimated parameters
		X_proj_obs = self.__aug_state_from_full_state__(X_proj)
		Xave_obs  = self.__aug_state_from_full_state__(Xave)
		
		# Matrix of weights and signs of the weights
		weights = np.sqrt( np.abs(self.W_c[:,0]) )
		signs   = np.sign( self.W_c[:,0] )
		
		# create matrix A that contains the error between the sigma points and the average
		A     = np.array([[]])
		i     = 0
		for x in X_proj_obs:
			
			error = signs[i]*weights[i]*(x - Xave_obs)
			
			# ignore when i==0, this will be done in the update
			if i==1:
				A = error.T
			elif i>1:
				A = np.hstack((A,error.T))
			i    += 1
		
		# put on the side the matrix sqrtQ, that have to be modified to fit the dimension of the augmenets state	
		new_sqrtQ = self.__new_Q__(sqrtQ)
		A = np.hstack((A,new_sqrtQ))
		
		# QR factorization
		q,L = np.linalg.qr(A.T)
		
		# execute Cholesky update
		x = signs[0]*weights[0]*(X_proj_obs[0,] - Xave_obs)
		
		L = self.chol_update(L, x.T, self.W_c[:,0])
		
		return L
		
	def compute_S_y(self, Z_proj, Zave, sqrtR):
		"""
		This function computes the squared covariance matrix using QR decomposition + a Cholesky update
		"""
		# Matrix of weights and signs of the weights
		weights = np.sqrt( np.abs(self.W_c[:,0]) )
		signs   = np.sign( self.W_c[:,0] )
		
		# create matrix A that contains the error between the sigma points outputs and the average
		A     = np.array([[]])
		i     = 0
		for z in Z_proj:
			error = signs[i]*weights[i]*(z - Zave)
			if i==1:
				A = error.T
			elif i>1:
				A = np.hstack((A,error.T))
			i    += 1
			
		# put the square root R matrix on the side
		A = np.hstack((A,sqrtR))
		
		# QR factorization
		q,L = np.linalg.qr(A.T)

		# NOW START THE CHOLESKY UPDATE
		z = signs[0]*weights[0]*(Z_proj[0,] - Zave)
		
		L = self.chol_update(L, z.T, self.W_c[:,0])
		
		return L
	
	def chol_update(self, L, X, W):
		"""
		This function computes the Cholesky update
		"""
		Lc = L.copy()
		signs   = np.sign( W )
	
		# NOW START THE CHOLESKY UPDATE
		# DO IT FOR EACH COLUMN IN THE X MATRIX
		
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
	
	def compute_C_x_x(self, X_next, X_now):
		"""
		This function computes the state-state cross covariance matrix (between the old Xold and the new Xnew state vectors).
		This is used by the smoothing process
		"""
		W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)
		Xave_next = self.average_proj(X_next)
		Xave_now  = self.average_proj(X_now)
		
		Vnext = self.__aug_state_from_full_state__(X_next - Xave_next)
		Vnow  = self.__aug_state_from_full_state__(X_now - Xave_now)
	
		Cxx = np.dot(np.dot(Vnext.T, W), Vnow)
		return Cxx

	def ukf_step(self, x, sqrtP, sqrtQ, sqrtR, t_old, t, z = None, verbose=False):
		"""
		z,x,S,sqrtQ,sqrtR,u_old,u,
		
		This methods contains all the steps that have to be performed by the UKF:
		
		1- prediction
		2- correction and update
		"""
		
		pars = x[self.n_state_obs:]
		x = x[:self.n_state_obs]
		
		# the list of sigma points (each sigma point can be an array, containing the state variables)
		# x, pars, sqrtP, sqrtQ = None, sqrtR = None
		Xs      = self.compute_sigma_points(x, pars, sqrtP, sqrtQ, sqrtR)
		
		if verbose:
			print "Sigma point Xs"
			print Xs
	
		# compute the projected (state) points (each sigma points is propagated through the state transition function)
		X_proj, Z_proj, Xfull_proj, Zfull_proj = self.sigma_point_proj(Xs,t_old,t)
		
		if verbose:
			print "Projected sigma points"
			print X_proj
	
		# compute the average
		Xave = self.average_proj(X_proj)
		Xfull_ave = self.average_proj(Xfull_proj)
		
		if verbose:
			print "Averaged projected sigma points"
			print Xave
		
		if verbose:
			print "Averaged projected full state"
			print Xfull_ave
		
		# compute the new squared covariance matrix S
		Snew = self.compute_S(X_proj,Xave,sqrtQ)
		
		if verbose:
			print "New squared S matrix"
			print Snew
		
		# redraw the sigma points, given the new covariance matrix
		x    = Xave[0,0:self.n_state_obs]
		pars = Xave[0,self.n_state_obs:]
		Xs   = self.compute_sigma_points(x, pars, Snew, sqrtQ, sqrtR)
		
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
		CovXZ = self.compute_cov_x_z(X_proj, Xave, Z_proj, Zave)
		
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
		X_corr = Xave + np.dot(K,z.reshape(self.n_outputs,1)-Zave.T).T
		
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
	
	@staticmethod
	def find_closest_matches(start, stop, time):
		"""
		Given the vector time and the start and stop values, the function returns the elements
		in the vector time that are as close as possible to the start and stop.
		NOTE:
		It is assumed that the vector time is sorted.
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
			
	
	def filter(self, start, stop, verbose = False, forSmoothing = False):
		"""
		This method starts the filtering process and performs a loop of ukf-steps
		"""
		# Read the output measured data
		measuredOuts = self.model.get_measured_output_data_series()
		
		# Get the time vector 
		time = pd.to_datetime(measuredOuts[:,0])
		
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
			
			# execute a filtering step
			try:
				X_corr, sP, Zave, S_y, Zfull_ave, X_full = self.ukf_step(x[i-1-ix_start], sqrtP[i-1-ix_start], sqrtQ, sqrtR, t_old, t, z, verbose=verbose)
			except Exception, e:
				print "Exception while running UKF step from {0} to {1}".format(t_old, t)
				print "The state X is"
				print x[i-1-ix_start]
				print "The sqrtP matrix is"
				print sqrtP[i-1-ix_start]
				raise Exception("Problem while performing a UKF step")
				
				
			x.append(X_corr)
			sqrtP.append(sP)
			y.append(Zave)
			y_full.append(Zfull_ave)
			Sy.append(S_y)
			x_full.append(X_full)
		
		# The first of the overall output vector is missing, copy from the second element
		y_full[0] = y_full[1]
		
		if forSmoothing:
			return time[ix_start:ix_stop], x, sqrtP, y, Sy, y_full, x_full, sqrtQ, sqrtR
		else:
			return time[ix_start:ix_stop], x, sqrtP, y, Sy, y_full
	
	def filter_and_smooth(self, start, stop, verbose=False):
		"""
		This method executes the filter and then the smoothing of the data
		"""
		# Run the filter
		time, X, sqrtP, y, Sy, y_full, x_full, sqrtQ, sqrtR = self.filter(start, stop, verbose = False, forSmoothing = True)
		
		print "SMOOTHING "*4
		
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
		# From point i with an estimation Xave[i], and S[i]
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
			Xs_i      = self.compute_sigma_points(x, pars, S_i, sqrtQ, sqrtR)
			
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
			Xave_plus_1 = self.average_proj(X_plus_1)
			
			if verbose:
				print "Averaged propagated sigma points"
				print Xave_plus_1
			
			# compute the new covariance matrix
			Snew = self.compute_S(X_plus_1, Xave_plus_1, sqrtQ)
			
			if verbose:
				print "Former S matrix used to draw the points"
				print S_i
				print "New Squared covaraince matrix"
				print Snew
			
			# compute the cross covariance matrix of the two states
			# (new state already corrected, coming from the "future", and the new just computed through the projection)
			Cxx  = self.compute_cov_x_x(X_plus_1, Xave_plus_1, Xs_i, Xs_i_ave)
			
			if verbose:
				print "Cross state-state covariance matrix"
				print Cxx
			
			# gain for the back propagation
			firstDivision = np.linalg.lstsq(Snew.T, Cxx.T)[0]
			D             = np.linalg.lstsq(Snew, firstDivision)[0]
			#D             = D.T
			
			correction = np.dot(np.matrix(Xsmooth[i+1]) - Xave_plus_1, D)
			if verbose:
				print "Old state"
				print X[i]
				print "Error:"
				print Xsmooth[i+1] - Xave_plus_1
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
	
	def parameter_estimation(self, start, stop, maxIter=100, verbose=False):
		"""
		This method provides a parameter estimation using the UKF smoother
		"""
		pars = []
		
		for i in range(maxIter):
			print "ITERATION "*3,i
			time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth = self.filter_and_smooth(start=start, stop=stop, verbose=verbose)
			
			x  = np.array(x)
			xs = np.array(Xsmooth)
			Ss = np.array(Ssmooth)
			
			x0 = xs[0,:self.n_state_obs]
			p0 = x[0,self.n_state_obs:]
			p  = xs[0,self.n_state_obs:]
			p_mean  = np.mean(xs[:,self.n_state_obs:], 0)
			print "init_pars=",p0
			print "pars=",p
			#print "pars_m=",p_mean
			pars.append(p)
			
			j = 0
			for var in self.model.get_variables():
				var.set_initial_value(xs[0,j])
				var.set_covariance(Ss[0,j,j])
				j += 1
			
			self.model.set_parameters_selected(p)
			j = 0
			for var in self.model.get_parameters():
				#var.SetInitialValue(p[j])
				#var.SetInitialValue(p_mean[j])
				var.set_covariance(Ss[0,j,j])
				j += 1
		
		return pars
			
			
			
			
		
			
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
		# From point i with an estimation Xave[i], and S[i]
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
			Xs_i        = self.compute_sigma_points(x_i,S_i)
			
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
			Xave_plus_1 = self.average_proj(x_plus_1)
			
			if verbose:
				print "Averaged propagated sigma points"
				print Xave_plus_1
			
			# compute the new covariance matrix
			Snew = self.compute_S(x_plus_1,Xave_plus_1,sqrtQ)
			
			if verbose:
				print "New Squared covaraince matrix"
				print Snew
			
			# compute the cross covariance matrix of the two states
			# (new state already corrected, coming from the "future", and the new just computed through the projection)
			Cxx  = self.compute_C_x_x(x_plus_1,Xave_plus_1,Xs_i,Xs_i_ave)
			
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
				print Xsmooth[i+1,0:self.n_state_obs] - Xave_plus_1[0,0:self.n_state_obs]
				print "Correction:"
				print np.dot(D, Xsmooth[i+1,0:self.n_state_obs] - Xave_plus_1[0,0:self.n_state_obs])
				
			# correction (i.e. smoothing, of the state estimation and covariance matrix)
			Xsmooth[i,self.n_state_obs:]   = Xhat[i,self.n_state_obs:]
			Xsmooth[i,0:self.n_state_obs]  = Xhat[i,0:self.n_state_obs] + np.dot(D, Xsmooth[i+1,0:self.n_state_obs] - Xave_plus_1[0,0:self.n_state_obs])
			
			# How to introduce constrained estimation
			Xsmooth[i,0:self.n_state_obs]  = self.constrained_state(Xsmooth[i,0:self.n_state_obs])
			
			if verbose:
				print "New smoothed state"
				print Xsmooth[i,:]
				raw_input("?")
			
			V              = np.dot(D,Ssmooth[i+1,:,:] - Snew)
			Ssmooth[i,:,:] = self.chol_update(S[i,:,:],V,-1*np.ones(self.n_state_obs))
			
		return (Xsmooth, Ssmooth)