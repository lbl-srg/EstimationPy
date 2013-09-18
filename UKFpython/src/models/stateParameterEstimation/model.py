import numpy as np
from ukf.Model import Model

class model(Model):
	
	# parameters of the chiler model
	a  = -0.8
	dt  = 0.1
	pars = (a, dt)
	
	# measurement covariance noise
	R     = np.array([[0.1]])
	sqrtR = np.linalg.cholesky(R)
	
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
	def __init__(self,a = -0.8, dt = 0.1, X0 = np.array([25.0, 25.0])):
		# use the initialization method of inherited class
		Model.__init__(self, 2, 1, 1, X0, None, [0.1], [1.0, 0.5])
		
		# set parameters of the model
		self.setPars(a, dt)

	"""
	This method modify the parameters of the model
	"""
	def setPars(self, a, dt):
		# parameters of the model
		self.pars = (a, dt)

	"""
	This method initialize the sampling step
	"""
	def setDT(self, dt = 0.1):
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
		a, dt = self.pars

		x_old = x[0]
		u     = u[0]

		# the simulated model contains a variation in the parameter b
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
		# return the output
		return np.array([x[0]])
