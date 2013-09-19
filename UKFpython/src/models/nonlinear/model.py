import numpy as np
from ukf.Model import Model

class model(Model):
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
	def __init__(self, a = -1.0, b = 0.5, dt = 0.1, X0 = np.array([1.0])):
		# use the initialization method of inherited class
		Model.__init__(self, 1, 1, 1, X0, None, [1.0], [1.0])
		
		# set parameters of the model
		self.setPars(a, b, dt)

	"""
	This method modify the parameters of the model
	"""
	def setPars(self, a, b, dt):
		# parameters of the model
		self.pars = (a, b, dt)

	"""
	This method initialize the sampling step
	"""
	def setDT(self,dt=0.1):
		self.dt = dt

	"""
	State evolution function

	* x: state vector at time t
	* u: input vector at time t
	* t: time

	it returns

	* new state at time t+1
	"""
	def functionF(self, val):
		(x,u_old,u,t_old,t,simulate) = val
		
		# parameters of the function
		(a, b, dt) = self.pars
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
	def functionG(self,x,u,t,simulate=True):
		# parameters of the function
		a, b, dt = self.pars
		x_old = x[0]
	
		# output values
		y  = b*np.sin(x_old)

		# return the output
		return np.array([y])
