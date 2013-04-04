import numpy as np

class model():
	# parameters of the model
	a  = -1.0
	b  = 0.5
	dt = 0.1 
	pars = (a, b, dt)
	
	# measurement covariance noise
	R     = np.array([[1.0]])
	sqrtR = np.linalg.cholesky(R)
	 
	# initial state vector (position, velocity)
	X0 = np.array([1.0])

	# process noise
	Q     = np.array([[1.0]])
	sqrtQ = np.linalg.cholesky(Q)

	# initialize state and output vectors
	n_states  = 1
	n_outputs = 1

	"""
	Initialize the model with parameters
	"""
	def __init__(self,a=-1.0, b=0.5, dt = 0.1, X0 = np.array([1.0])):
		# parameters of the model
		self.a = a
		self.b = b
		self.dt = dt
		self.pars = (a, b, dt)
	
		# measurement covariance noise
		self.R     = np.array([[0.004]])
		self.sqrtR =  np.linalg.cholesky(self.R)
		 
		# initial state vector (position, velocity)
		self.X0 = X0

		# process noise
		self.Q     = np.array([[0.01]])
		self.sqrtQ = np.linalg.cholesky(self.Q)
	
		# initialize state and output vectors
		n_states  = 1
		n_outputs = 1

	"""
	This method define the initial state
	"""
	def setInitialState(self,X0):
		self.X0 = X0	

	"""
	This method initialize the sampling step
	"""
	def setDT(self,dt=0.1):
		self.dt = dt

	"""
	This method initialize the state noise matrix
	"""
	def setQ(self,Q):
		self.Q     = Q
		self.sqrtQ = np.linalg.cholesky(Q)

	"""
	This method initialize the measurement noise matrix
	"""
	def setR(self,R):
		self.R     = R
		self.sqrtR = np.linalg.cholesky(R)		

	"""
	State evolution function

	* x: state vector at time t
	* u: input vector at time t
	* t: time

	it returns

	* new state at time t+1
	"""
	def functionF(self,x,u,t):
		# parameters of the function
		a, b, dt = self.pars
		x_old = x[0]
	
		# new state value
		x_new = a*np.sin(x_old)

		# return the state
		return np.array([x_new])


	"""
	Output measurement function

	* x: state vector at time t
	* u: input vector at time t
	* t: time

	it returns

	* output at time t
	"""
	def functionG(self,x,u,t):
		# parameters of the function
		a, b, dt = self.pars
		x_old = x[0]
	
		# output values
		y  = b*np.sin(x_old)

		# return the output
		return np.array([y])
	
	"""
	get the number of states variables
	"""
	def getNstates(self):
		return self.n_states

	"""
	get the number of states variables
	"""
	def getNoutputs(self):
		return self.n_outputs
