import numpy as np
from multiprocessing import Pool

def Function(val):
	"""
	This function takes as argument a tuple of elements. These elements are then used to compute the state
	transition function
	
	the tuple contains:
	
	* ** m **:     the model
	* ** x **:     the state vector
	* ** u_uld **: input vector at previous step
	* ** u ** :     input vector at actual step
	* ** t_old **: previous time step
	* ** t **:     actual time step
	* **other**: flags used for the simulation
	
	"""
	# Get the values from the tuple 
	(m,x,u_old,u,t_old,t,Flags,) = val
	
	# execute the state transition function associated to the model
	return m.functionF((x,u_old,u,t_old,t,False,))

class ukf():
	"""
	This class represents an Unscented Kalman Filter (UKF) that can be used for the state and parameter estimation of nonlinear dynamic systems
	"""
	# number of state variables, outputs and sigma points
	n_state     = 0
	n_state_obs = 0
	n_outputs   = 0
	n_points    = 0

	# parameters of the Unscented Kalman Filter
	alpha     = 0
	k         = 0
	beta      = 0
	lambd     = 0

	# weights of the UKF
	W_m       = np.zeros((2*n_state_obs + 1, 1))
	W_c       = np.zeros((2*n_state_obs + 1, 1))
	CstCov    = 0
	
	# constraints of the observed state
	ConstrHigh = np.empty(n_state_obs)
	ConstrHigh.fill(False)
	ConstrLow = np.empty(n_state_obs)
	ConstrLow.fill(False)
	# Max Value of the constraints
	ConstrValueHigh = np.zeros(n_state_obs)
	# Min Value of the constraints
	ConstrValueLow = np.zeros(n_state_obs)
	
	
	def __init__(self, n_state, n_state_obs, n_outputs):
		"""
		Initialization of the UKF and its parameters. The dynamic system is characterized by
		
		** n_state     -- state variables,
		** n_state_obs -- state variables that are observed (estimated)
		** n_outputs   -- output variables
		
		The initialization assign these parameters then,
		
		1- compute the number of sigma points to be used
		2- define the parameters of the filter
		3- compute the weights associated to each sigma point
		4- initialize the constraints on the observed state variables
		 
		"""
		# set the number of states variables and outputs
		self.n_state     = n_state
		self.n_state_obs = n_state_obs
		self.n_outputs   = n_outputs

		# compute the sigma points
		self.n_points    = 1 + 2*self.n_state_obs

		# define UKF parameters
		self.setUKFparams()

		# compute the weights
		self.computeWeights()
		
		# set the default constraints, not active
		self.ConstrHigh = np.empty(self.n_state_obs)
		self.ConstrHigh.fill(False)
		self.ConstrLow = np.empty(self.n_state_obs)
		self.ConstrLow.fill(False)
		
		# Max Value of the constraints
		self.ConstrValueHigh = np.zeros(self.n_state_obs)
		# Min Value of the constraints
		self.ConstrValueLow  = np.zeros(self.n_state_obs)

	
	def setDefaultUKFparams(self):
		"""
		This method set the default parameters of the UKF
		"""
		self.alpha    = 0.01
		self.k        = 0
		self.beta     = 2
		self.lambd    = (self.alpha**2)*(self.n_state_obs + self.k) - self.n_state_obs
		self.sqrtC    = self.alpha*np.sqrt(self.n_state_obs + self.k)

	
	def setUKFparams(self, alpha = 1.0/np.sqrt(3.0), beta = 2, k = None):
		"""
		This method set the non default parameters of the UKF
		"""
		self.alpha     = alpha
		self.beta      = beta

		if k == None:
			self.k = 3 - self.n_state_obs
		else:
			self.k = k
		
		self.lambd    = (self.alpha**2)*(self.n_state_obs + self.k) - self.n_state_obs
		self.sqrtC    = self.alpha*np.sqrt(self.k + self.n_state_obs)		

	
	def setHighConstraints(self,flag,values):
		"""
		This method imposes the upper constraints of the observed state variables.
		"""
		self.ConstrHigh      = flag
		self.ConstrValueHigh = values

	def setLowConstraints(self,flag,values):
		"""
		This method imposes the lower constraints of the observed state variables.
		"""
		self.ConstrLow      = flag
		self.ConstrValueLow = values

	
	def computeWeights(self):
		"""
		This method computes the weights of the UKF filter. These weights are associated to each sigma point and are used to
		compute the mean value (W_m) and the covariance (W_c) of the estimation
		"""
		self.W_m       = np.zeros((1+self.n_state_obs*2,1))
		self.W_c       = np.zeros((1+self.n_state_obs*2,1))
		
		self.W_m[0,0]  = self.lambd/(self.n_state_obs + self.lambd)
		self.W_c[0,0]  = self.lambd/(self.n_state_obs + self.lambd) + (1 - self.alpha**2 + self.beta)

		for i in range(2*self.n_state_obs):
			self.W_m[i+1,0] = 1.0/(2.0*(self.n_state_obs + self.lambd))
			self.W_c[i+1,0] = 1.0/(2.0*(self.n_state_obs + self.lambd))

	def squareRoot(self,A):
		"""
		This method computes the square root of a square matrix A, using the Cholesky factorization
		"""
		try:
			sqrtA = np.linalg.cholesky(A)
			return sqrtA

		except np.linalg.linalg.LinAlgError:
			print "Matrix is not positive semi-definite"
			print A
			raw_input("press...")
			return A	
	
	def constrainedState(self,X):
		"""
		This method apply the constraints to the state vector (only to the estimated states)
		"""		
		# Check for every observed state
		for i in range(self.n_state_obs):
		
			# if the constraint is active and the threshold is violated
			if self.ConstrHigh[i] and X[i] > self.ConstrValueHigh[i]:
				X[i] = self.ConstrValueHigh[i]
				
			# if the constraint is active and the threshold is violated	
			if self.ConstrLow[i] and X[i] < self.ConstrValueLow[i]:
				X[i] = self.ConstrValueLow[i]
		
		return X
				
	def computeSigmaPoints(self,x,sqrtP):
		"""
		This method computes the sigma points, Its inputs are
		
		* x     -- the state vector around the points will be propagated,
		* sqrtP -- the square root matrix of the covariance P, that is used to spread the points
		
		"""
	
		# reshape the state vector
		x = x.reshape(1,self.n_state)

		# initialize the matrix of sigma points
		# the result is
		# [[0.0, 0.0, 0.0],
		#  [0.0, 0.0, 0.0],
		#  [0.0, 0.0, 0.0],
		#  [0.0, 0.0, 0.0],
		#      ....
		#  [0.0, 0.0, 0.0]]
		Xs      = np.zeros((self.n_points,self.n_state))

		# Now using the sqrtP matrix that is lower triangular, I create the sigma points
		# by adding and subtracting the rows of the matrix sqrtP, to the lines of Xs
		# [[s11, 0  , 0  ],
		#  [s12, s22, 0  ],
		#  [s13, s23, s33]]
		i = 1
		Xs[0,:] = x
		
		for row in sqrtP:
			Xs[i,:]									  = x
			Xs[i+self.n_state_obs,:]				  = x
			
			Xs[i,0:self.n_state_obs]                  = x[0,0:self.n_state_obs] + self.sqrtC*row
			Xs[i+self.n_state_obs,0:self.n_state_obs] = x[0,0:self.n_state_obs] - self.sqrtC*row
			
			# TODO:
			# How to introduce constrained points
			# Xs[i,0:self.n_state_obs]                  = self.constrainedState(Xs[i,0:self.n_state_obs])
			# Xs[i+self.n_state_obs,0:self.n_state_obs] = self.constrainedState(Xs[i+self.n_state_obs,0:self.n_state_obs])
			
			i += 1
		
		return Xs

	def sigmaPointProj(self,m,Xs,u_old,u,t_old,t):
		"""
		This function, given a set of sigma points Xs, propagate them using the state transition function.
		The simulations are run in parallel
		"""
		# initialize the vector of the NEW STATES
		X_proj = np.zeros((self.n_points,self.n_state))
		# this flag enables to run the simulation in parallel
		parallel = False
		
		if parallel: 
			# execute each state projection in parallel
			pool = Pool()
			res = pool.map_async(Function, ((m,x,u_old,u,t_old,t,False,) for x in Xs))
			pool.close()
			pool.join()
			
			# collect the results of the simulations
			j = 0
			for X in res.get():
				X_proj[j,:] = X
				j += 1
		else:
			j = 0
			for x in Xs:
				values = (x, u_old, u, t_old, t, True)
				X_proj[j,:] = m.functionF(values)
				j += 1
				
		return X_proj

	
	def sigmaPointOutProj(self,m,Xs,u,t):
		"""
		This function computes the outputs of the model, given a set of sigma points Xs as well as inputs u and time step t
		"""
		# initialize the vector of the outputs
		Z_proj = np.zeros((self.n_points,self.n_outputs))
		j = 0
		for x in Xs:
			Z_proj[j,:] = m.functionG(x,u,t,False)
			j += 1
		return Z_proj

	def averageProj(self,X_proj):
		"""
		This function averages the projection of the sigma points using the weights vector W_m
		"""
		# make sure that the shape is [1+2*n, n]
		X_proj.reshape(self.n_points, self.n_state)
		
		# dot product of the two matrices
		avg = np.dot(self.W_m.T, X_proj)
		
		return avg
	
	def averageOutProj(self,Z_proj):
		"""
		This function averages the outputs of the sigma points using the weights vector W_m
		"""
		# make sure that the shape is [1+2*n, n]
		Z_proj.reshape(self.n_points, self.n_outputs)
		
		# dot product of the two matrices
		avg = np.dot(self.W_m.T, Z_proj)
		return avg

	def computeP(self,X_p,Xa,Q):
		"""
		This function computes the state covariance matrix P as
		
		P[i,j] = W_c[i]*(Xs[i] - Xavg)^2 + Q[i,j]
		
		"""
		W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)

		V = np.zeros(X_p.shape)
		for j in range(self.n_points):
			V[j,:]   = X_p[j,:] - Xa[0]
		
		Pnew = np.dot(np.dot(V.T,W),V) + Q
		return Pnew

	def computeCovZ(self,Z_p,Za,R):
		"""
		This function computes the output covariance matrix CovZ as
		
		CovZ[i,j] = W_c[i]*(Zs[i] - Zavg)^2 + R[i,j]
		
		"""
		W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)

		V =  np.zeros(Z_p.shape)
		for j in range(self.n_points):
			V[j,:]   = Z_p[j,:] - Za[0]
		
		CovZ = np.dot(np.dot(V.T,W),V) + R
		return CovZ

	
	def computeCovXZ(self,X_p, Xa, Z_p, Za):
		"""
		This function computes the state-output cross covariance matrix (between X and Z)
		"""
		W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)
		
		# Old version
		# Vx = np.zeros(X_p.shape)
		Vx = np.zeros((self.n_points,self.n_state_obs))
		for j in range(self.n_points):
			Vx[j,:]   = X_p[j,0:self.n_state_obs] - Xa[:,0:self.n_state_obs]	
		
		Vz = np.zeros(Z_p.shape)
		for j in range(self.n_points):
			Vz[j,:]   = Z_p[j,:] - Za[0]
	
		CovXZ = np.dot(np.dot(Vx.T,W),Vz)
		return CovXZ
	
	def computeS(self,X_proj,Xave,sqrtQ):
		"""
		This function computes the squared covariance matrix using QR decomposition + a Cholesky update
		"""
		# take the first part of the state vector
		# The observed states
		X_proj_obs = X_proj[:,0:self.n_state_obs]
		
		Xave_obs  = Xave[:,0:self.n_state_obs]
		
		weights = np.sqrt( np.abs(self.W_c[:,0]) )
		signs   = np.sign( self.W_c[:,0] )
		
		A     = np.array([[]])
		i     = 0
		for x in X_proj_obs:
			error = signs[i]*weights[i]*(x - Xave_obs)
			
			if i==1:
				A = error.T
			elif i>1:
				A = np.hstack((A,error.T))
			i    += 1
			
		
		A = np.hstack((A,sqrtQ))
		
		q,L = np.linalg.qr(A.T,mode='full')
		
		# NOW START THE CHOLESKY UPDATE
		x = signs[0]*weights[0]*(X_proj_obs[0,] - Xave_obs)
		
		L = self.cholUpdate(L,x.T,self.W_c[:,0])
		
		return L
		
	
	def computeSy(self,Z_proj,Zave,sqrtR):
		"""
		This function computes the squared covariance matrix using QR decomposition + a Cholesky update
		"""
		weights = np.sqrt( np.abs(self.W_c[:,0]) )
		signs   = np.sign( self.W_c[:,0] )
		
		A     = np.array([[]])
		i     = 0
		for z in Z_proj:
			error = signs[i]*weights[i]*(z - Zave)
			if i==1:
				A = error.T
			elif i>1:
				A = np.hstack((A,error.T))
			i    += 1
		A = np.hstack((A,sqrtR))
		
		q,L = np.linalg.qr(A.T,mode='full')

		# NOW START THE CHOLESKY UPDATE
		z = signs[0]*weights[0]*(Z_proj[0,] - Zave)
		
		L = self.cholUpdate(L,z.T,self.W_c[:,0])
		
		return L
	
	
	def cholUpdate(self,L,X,W):
		"""
		This function computes the Cholesky update
		"""
		L = L.copy()
		weights = np.sqrt( np.abs( W ) )
		signs   = np.sign( W )
	
		# NOW START THE CHOLESKY UPDATE
		# DO IT FOR EACH COLUMN IN THE X MATRIX
		
		(row, col) = X.shape
		
		for j in range(col):
			x = X[0:,j]
			
			for k in range(row):
				rr_arg    = L[k,k]**2 + signs[0]*x[k]**2
				rr        = 0.0 if rr_arg < 0 else np.sqrt(rr_arg)
				c         = rr / L[k,k]
				s         = x[k] / L[k,k]
				L[k,k]    = rr
				L[k,k+1:] = (L[k,k+1:] + signs[0]*s*x[k+1:])/c
				x[k+1:]   = c*x[k+1:]  - s*L[k, k+1:]
				
		return L
	
	
	def computeCxx(self,X_next,Xave_next,X_now,Xave_now):
		"""
		This function computes the state-state cross covariance matrix (between the old Xold and the new Xnew state vectors).
		This is used by the smoothing process
		"""
		W = np.diag(self.W_c[:,0]).reshape(self.n_points,self.n_points)
		
		Vnext = np.zeros((self.n_points,self.n_state_obs))
		for j in range(self.n_points):
			Vnext[j,:]   = X_next[j,0:self.n_state_obs] - Xave_next[:,0:self.n_state_obs]	
		
		Vnow = np.zeros((self.n_points,self.n_state_obs))
		for j in range(self.n_points):
			Vnow[j,:]   = X_now[j,0:self.n_state_obs] - Xave_now[:,0:self.n_state_obs]
	
		Cxx = np.dot(np.dot(Vnext.T,W),Vnow)
		return Cxx

	
	def ukf_step(self,z,x,S,sqrtQ,sqrtR,u_old,u,t_old,t,m,verbose=False):		
		"""
		This methods contains all the steps that have to be performed by the UKF:
		
		1- prediction
		2- correction and update
		
		"""
		
		# the list of sigma points (each signa point can be an array, containing the state variables)
		Xs      = self.computeSigmaPoints(x,S)
	
		if verbose:
			print "Sigma point Xs"
			print Xs
	
		# compute the projected (state) points (each sigma points is propagated through the state transition function)
		X_proj = self.sigmaPointProj(m,Xs,u_old,u,t_old,t)
		
		if verbose:
			print "Projected sigma points"
			print X_proj
	
		# compute the average
		Xave = self.averageProj(X_proj)
		
		if verbose:
			print "Averaghed projected sigma points"
			print Xave
		
		# compute the new squared covariance matrix S
		Snew = self.computeS(X_proj,Xave,sqrtQ)
		
		if verbose:
			print "New squared S matrix"
			print Snew
		
		# redraw the sigma points, given the new covariance matrix
		Xs      = self.computeSigmaPoints(Xave,Snew)
		
		# keep the last part of the projection for the other states not updated
		Xs[:,self.n_state_obs:] = X_proj[:,self.n_state_obs:]
		
		if verbose:
			print "New sigma points"
			print Xs

		# compute the projected (outputs) points (each sigma points is propagated through the output function, this should not require a simulation,
		# just the evaluation of a function since the output can be directly computed if the state vector and inputs are known )
		Z_proj = self.sigmaPointOutProj(m,Xs,u,t)
		
		if verbose:
			print "Output projection of new sigma points"
			print Z_proj

		# compute the average output
		Zave = self.averageOutProj(Z_proj)
		
		if verbose:
			print "Averaged output projection of new sigma points"
			print Zave

		# compute the innovation covariance (relative to the output)
		Sy = self.computeSy(Z_proj,Zave,sqrtR)
		
		if verbose:
			print "Output squared covariance matrix"
			print Sy
		
		# compute the cross covariance matrix
		CovXZ = self.computeCovXZ(X_proj, Xave, Z_proj, Zave)
		
		if verbose:
			print "State output covariance matrix"
			print CovXZ
	
		# Data assimilation step
		# The information obtained in the prediction step are corrected with the information
		# obtained by the measurement of the outputs
		# In other terms, the Kalman Gain (for the correction) is computed
		firstDivision = np.linalg.lstsq(Sy.T,CovXZ.T)[0]
		K             = np.linalg.lstsq(Sy, firstDivision)[0]
		K             = K.T
		
		# State correction using the measurements
		X_corr = Xave
		X_corr[:,0:self.n_state_obs] = Xave[:,0:self.n_state_obs] + np.dot(K,z.reshape(self.n_outputs,1)-Zave.T).T
		
		# If constraints are active, they are imposed in order to avoid the corrected value to fall outside
		X_corr[0,0:self.n_state_obs] = self.constrainedState(X_corr[0,0:self.n_state_obs])
		
		if verbose:
			print "New state corrected"
			print X_corr
			raw_input("?")
		
		# The covariance matrix is corrected too
		U      = np.dot(K,Sy)
		S_corr = self.cholUpdate(Snew,U,-1*np.ones(self.n_state))

		return (X_corr, S_corr, Zave, Sy)
	
	
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
			Xs_i        = self.computeSigmaPoints(x_i,S_i)
			
			if verbose:
				print "Sigma points"
				print Xs_i
			
			# mean of the sigma points
			Xs_i_ave    = self.averageProj(Xs_i)
			
			if verbose:
				print "Mean of the sigma points"
				print Xs_i_ave
			
			# propagate the sigma points
			x_plus_1    = self.sigmaPointProj(m,Xs_i,U[i],U[i+1],time[i],time[i+1])
			
			if verbose:
				print "Propagated sigma points"
				print x_plus_1
			
			# average of the sigma points
			Xave_plus_1 = self.averageProj(x_plus_1)
			
			if verbose:
				print "Averaged propagated sigma points"
				print Xave_plus_1
			
			# compute the new covariance matrix
			Snew = self.computeS(x_plus_1,Xave_plus_1,sqrtQ)
			
			if verbose:
				print "New Squared covaraince matrix"
				print Snew
			
			# compute the cross covariance matrix of the two states
			# (new state already corrected, coming from the "future", and the new just computed through the projection)
			Cxx  = self.computeCxx(x_plus_1,Xave_plus_1,Xs_i,Xs_i_ave)
			
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
			Xsmooth[i,0:self.n_state_obs]  = self.constrainedState(Xsmooth[i,0:self.n_state_obs])
			
			if verbose:
				print "New smoothed state"
				print Xsmooth[i,:]
				raw_input("?")
			
			V              = np.dot(D,Ssmooth[i+1,:,:] - Snew)
			Ssmooth[i,:,:] = self.cholUpdate(S[i,:,:],V,-1*np.ones(self.n_state_obs))
			
		return (Xsmooth, Ssmooth)

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
			Xs[i,:]					= x
			Xs[i+self.n_augState,:]	= x
			
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
			X_proj[j,:] 		      = X
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
		S_1   = np.hstack((S, 									     		np.zeros((self.n_state_obs, self.n_state_obs)),	np.zeros((self.n_state_obs, self.n_outputs))))
		S_2   = np.hstack((np.zeros((self.n_state_obs, self.n_state_obs)),  actualSq,						      	    	np.zeros((self.n_state_obs, self.n_outputs))))
		S_3   = np.hstack((np.zeros((self.n_outputs, self.n_state_obs)), 	np.zeros((self.n_outputs, self.n_state_obs)), 	Sr))
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