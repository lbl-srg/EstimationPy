import numpy as np
from ukf.Model import Model

class model(Model):
	# parameters of the model
	Kv = 1.0
	Kx = 0.5
	M  = 1.0
	dt = 0.1
	pars = (Kv, Kx, M, dt)
	
	# measurement covariance noise
	R     = np.array([[1.0,    0.0,   0.0],
			  [0.0,    1.0,   0.0],
			  [0.0,    0.0,   1.0]])
	sqrtR = np.linalg.cholesky(R)
	
	# initial state vector (position, velocity)
	X0 = np.array([1.0, 0.0])

	# process noise
	Q     = np.array([[1.0, 0.0],
			  [0.0, 1.0]])
	sqrtQ = np.linalg.cholesky(Q)

	# initialize state and output vectors
	n_states  = 2
	n_outputs = 3

	"""
	Initialize the model with parameters
	"""
	def __init__(self,Kv = 1.0, Kx = 0.5, M = 1.0, dt = 0.1, X0 = np.array([1.0, 0.0])):
		# use the initialization method of inherited class
		Model.__init__(self, 2, 2, 3, X0, None, [1.0, 1.0, 1.0], [1.0, 1.0])
		
		# set parameters of the model
		self.setPars(Kv, Kx, M, dt)
		
	"""
	This method modify the parameters of the model
	"""
	def setPars(self, Kv, Kx, M, dt):
		# parameters of the model
		self.pars = (Kv, Kx, M, dt)

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
		Kv, Kx, M, dt = self.pars
		x_old = x[0]
		v_old = x[1]
		F_old = u[0]
	
		# new state value
		x_new = x_old + v_old*dt
		v_new = v_old + dt/M*(F_old -Kv*v_old - Kx*x_old)

		# return the state
		return np.array([x_new, v_new])

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
		Kv, Kx, M, dt = self.pars
	
		# output values
		vel  = x[1]
		acc  = u[0] - Kv*x[1] - Kx*x[0]
		ekin = 0.5*M*vel**2

		# return the output
		return np.array([vel, acc, ekin])