import numpy as np

class model():
	# parameters of the chiler model
	cp  = 4186
	Mch = 50
	Mcw = 90
	wch = 5
	wcw = 8
	dTch_nom = 8
	dTcw_nom = 5
	W_nom    = 50000

	dt  = 0.1
	pars = (cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, dt)
	
	# measured outputs are: Tch, Tcw
	# measurement covariance noise
	R     = np.array([[1.0,    0.0],
			  [0.0,    1.0]])
	sqrtR = np.linalg.cholesky(R)
	
	# the states of the system are (Tch, Tcw, COP)
	# initial state vector is
	X0 = np.array([25.0, 25.0, 5.5])

	# process noise
	Q     = np.array([[1.0, 0.0, 0.0],
			  [0.0, 1.0, 0.0],
			  [0.0, 0.0, 0.1]])
	sqrtQ = np.linalg.cholesky(Q)

	# initialize state and output vectors
	n_states  = 3
	n_outputs = 2

	"""
	Initialize the model with parameters
	"""
	def __init__(self,cp=4186, Mch=50, Mcw=90, wch=5, wcw=8, dTch_nom=8, dTcw_nom=5, W_nom=50000, dt=0.1, X0=np.array([25.0, 25.0])):
		# parameters of the model
		self.cp  = cp
		self.Mch = Mch
		self.Mcw = Mcw
		self.wch = wch
		self.wcw = wcw
		self.dt  = dt
		self.dTch_nom = dTch_nom
		self.dTcw_nom = dTcw_nom
		self.W_nom    = W_nom
		self.pars = (self.cp, self.Mch, self.Mcw, self.wch, self.wcw, self.dTch_nom, self.dTcw_nom, self.W_nom, self.dt)

		# measurement covariance noise
		self.R     = np.array([[2.0,    0.0],
			  	       [0.0,    2.0]])
		self.sqrtR =  np.linalg.cholesky(self.R)
		 
		# initial state vector (position, velocity)
		self.X0 = X0

		# process noise
		Q     = np.array([[1.0**2, 0.0, 0.0],
			  	  [0.0, 1.0**2, 0.0],
			  	  [0.0, 0.0, 0.2**2]])
		self.sqrtQ = np.linalg.cholesky(self.Q)
	
		# initialize state and output vectors
		n_states   = 3
		n_outputs  = 2

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
		cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, dt = self.pars

		Tch_old = x[0]
		Tcw_old = x[1]
		COP_old = x[2]
		Tch_in  = u[0]
		Tcw_in  = u[1]
		W_in    = u[2]
		
		(Pch, COP) = self.functionPch(x,u,t,simulate)

		# new state value
		Tch = dt/(Mch*cp)*(wch*cp*(Tch_in - Tch_old) - Pch ) + Tch_old
		Tcw = dt/(Mcw*cp)*(wcw*cp*(Tcw_in - Tcw_old) + Pch + W_in ) + Tcw_old

		# return the state
		return np.array([Tch, Tcw, COP])


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
		cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, dt = self.pars
	
		Tch     = x[0]
		Tcw     = x[1]

		# return the output
		return np.array([Tch, Tcw])
	
	"""
	This function computes the cooling power of the chiller
	"""
	def functionPch(self,x,u,t,simulate=True):
		# parameters of the function
		cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, dt = self.pars

		Tch = x[0]
		Tcw = x[1]
		
		Tch_in  = u[0]
		Tcw_in  = u[1]
		W_in    = u[2]
	
		# nonlinear function, the compressor characteristic
		dW     = np.absolute(W_in)/W_nom
		dCh    = (np.absolute(Tch - Tch_in)-dTch_nom)/dTch_nom
		dCh    = np.max([np.min([dCh, 1]),-1])
		dCw    = (np.absolute(Tcw - Tcw_in)-dTcw_nom)/dTcw_nom
		dCw    = np.max([np.min([dCh, 1]),-1])
		factor = np.max([np.min([1 -0.1*dW**2 -0.1*dCh**2 -0.1*dCw**2 , 1]),0.0])

		# simulation of a fault
		if t>=2000 and simulate:
			COP = 0.5
		else:
			COP = x[2]
				
		Pch = dW*COP*W_nom*factor

		# return the state
		return np.array([Pch, COP])

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
