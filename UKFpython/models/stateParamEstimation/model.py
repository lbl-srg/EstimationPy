import numpy as np

class model():
	# parameters of the chiler model
	a  = -0.8

	dt  = 0.1

	pars = (a, dt)
	
	# measured outputs are: Tch, Tcw
	# measurement covariance noise
	R     = np.array([[0.1]])
	sqrtR = np.linalg.cholesky(R)
	
	# the states of the system are (Tch, Tcw, COP)
	# initial state vector is
	X0 = np.array([1.0, 5.0])

	# process noise
	Q     = np.array([[0.1, 0.0],
		  	  [0.0, 0.5]])
	sqrtQ = np.linalg.cholesky(Q)

	# initialize state and output vectors
	n_states  = 2
	n_outputs = 1

	"""
	Initialize the model with parameters
	"""
	def __init__(self,a = -0.8, dt=0.1, X0=np.array([25.0, 25.0])):
		# parameters of the model
		self.a = a
		self.dt  = dt
		self.pars = (self.a, self.dt)

		# measurement covariance noise
		R     = np.array([[0.1]])
		self.sqrtR =  np.linalg.cholesky(self.R)
		 
		# initial state vector (position, velocity)
		self.X0 = X0

		# process noise
		Q     = np.array([[0.1, 0.0],
		  	  	  [0.0, 0.5]])
		self.sqrtQ = np.linalg.cholesky(self.Q)
	
		# initialize state and output vectors
		n_states   = 2
		n_outputs  = 1

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
	def functionF(self,x,u,t,simulate=True):
		# parameters of the function
		a, dt = self.pars

		x_old = x[0]
		u     = u[0]

		# the simulated model contains a variationin the parameter b
		# the non simulated model that try to estimate it not
		if not simulate:		
			b_old = x[1]
		else:
			if t >= 5.0 and t<=12.0:
				b_old = 10.0
			elif t>=12.0 and t<=20.0:
				b_old = 2.0
			else:
				b_old = x[1]
			
		# new state value
		x = x_old + dt*(a*(1+np.sin(x_old))*x_old + b_old*u)
	
		# return the state
		return np.array([x, b_old])


	"""
	Output measurement function

	* x: state vector at time t
	* u: input vector at time t
	* t: time

	it returns

	* output at time t
	"""
	def functionG(self,x,u,t,simulate=True):
		# parameters of the function
		a, dt = self.pars

		# return the output
		return np.array([x[0]])

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
