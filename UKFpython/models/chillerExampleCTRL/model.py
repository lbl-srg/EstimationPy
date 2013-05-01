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
	W_nom    = 80000
	
	Ti = 10
	Td = 5
	K  = -10
	CSmax = 1
	CSmin = 0
	b     = 1
	c     = 1
	N     = 8
	 
	dt  = 0.1
	DT  = 60.0
	
	pars = (cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT)
	
	# measured outputs are: 
	# Tch: temperature chilled water,
	# Tcw: temperature condenser water,
	# W:   compressor power consumption
	
	# measurement covariance noise
	R     = np.diag([1.0, 1.0, 200])
	sqrtR = np.linalg.cholesky(R)
	
	# the states of the system are (Tch, Tcw, COP, d, CS, Tch_sp)
	# initial state vector is
	X0 = np.array([25.0, 25.0, 5.0, 0, 0, 4.0])

	# process noise for the state variables that are of interest
	# Tch, Tcw, COP
	Q     = np.diag([1.0, 1.0, 0.1, 0.0001, 0.0001, 0.0001])
	sqrtQ = np.linalg.cholesky(Q)

	# initialize state and output vectors
	n_states    = 6
	n_statesTot = 6
	n_outputs   = 3

	"""
	Initialize the model with parameters
	"""
	def __init__(self,cp=4186, Mch=50, Mcw=90, wch=5, wcw=8, dTch_nom=8, dTcw_nom=5, W_nom=70000, Ti = 10.0, Td = 5.0, K  = -10.0, CSmax = 1.0, CSmin = 0.0, b = 1.0, c = 1.0, N = 8.0, dt=0.1, DT = 60.0, X0=np.array([25.0, 25.0, 5.5, 0, 0, 4.0])):
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
		self.Ti = Ti
		self.Td = Td
		self.K  = K
		self.CSmax = CSmax
		self.CSmin = CSmin
		self.b     = b
		self.c     = c
		self.N     = N
		self.DT    = DT
		
		self.pars = (self.cp, self.Mch, self.Mcw, self.wch, self.wcw, self.dTch_nom, self.dTcw_nom, self.W_nom, self.Ti, self.Td, self.K, self.N, self.b, self.c, self.CSmax, self.CSmin, self.dt, self.DT)

		# measurement covariance noise
		R     = np.diag([1.0, 1.0, 200])
		sqrtR = np.linalg.cholesky(R)
		 
		# initial state vector (position, velocity)
		self.X0 = X0

		# process noise
		Q     = np.diag([1.0, 1.0, 0.1, 0.0001, 0.0001, 0.0001])
		sqrtQ = np.linalg.cholesky(Q)
	
		# initialize state and output vectors
		n_states    = 6
		n_statesTot = 6
		n_outputs   = 3

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
	This method modify the parameter sof the model
	"""
	def setPars(self,cp=4186, Mch=50, Mcw=90, wch=5, wcw=8, dTch_nom=8, dTcw_nom=5, W_nom=70000, Ti = 10.0, Td = 5.0, K  = -10.0, CSmax = 1.0, CSmin = 0.0, b = 1.0, c = 1.0, N = 8.0):
		# parameters of the model
		self.cp  = cp
		self.Mch = Mch
		self.Mcw = Mcw
		self.wch = wch
		self.wcw = wcw
		self.dTch_nom = dTch_nom
		self.dTcw_nom = dTcw_nom
		self.W_nom    = W_nom
		self.Ti = Ti
		self.Td = Td
		self.K  = K
		self.CSmax = CSmax
		self.CSmin = CSmin
		self.b     = b
		self.c     = c
		self.N     = N
		
		self.pars = (self.cp, self.Mch, self.Mcw, self.wch, self.wcw, self.dTch_nom, self.dTcw_nom, self.W_nom, self.Ti, self.Td, self.K, self.N, self.b, self.c, self.CSmax, self.CSmin, self.dt, self.DT)


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
	def functionF_dt(self,x,u,t,simulate):
		# parameters of the function
		cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT = self.pars
		
		# state variable of interest
		Tch_old = x[0]
		Tcw_old = x[1]
		COP_old = x[2]
		
		# state variables for the controller
		D_old   = x[3]
		CS_old  = x[4]
		SP_old  = x[5]
		PV_old  = Tch_old
		
		# input variables
		Tch_in  = u[0]
		Tcw_in  = u[1]
		Tch_sp  = u[2]
		
		# COMPUTE THE CONTROL ACTION
		# with incremental PID controller
		dsp = Tch_sp - SP_old
		dpv = Tch_old - PV_old
		 
		dp      = K*(b*dsp - dpv)
		di      = K*dt/Ti*( Tch_sp - Tch_old)
		if Td > 0:
			d   = (Td*D_old + K*N*Td*(c*dsp - dpv))/(Td + N*dt);
		else:
			d   = (Td*D_old + K*N*Td*(c*dsp - dpv));	
		
		dd      = d - D_old
		dcs     = dp + di + dd
		cs      = CS_old + dcs
		
		# Anti Windup
		if cs >= CSmax and simulate:
			cs = CSmax
			
		if cs <= CSmin and simulate:
			cs = CSmin 
		
		# Get the cooling POwer and the COP
		(Pch, COP) = self.functionPch(x,u,cs,t,simulate)

		# new state value
		Tch = dt/(Mch*cp)*(wch*cp*(Tch_in - Tch_old) - Pch ) + Tch_old
		Tcw = dt/(Mcw*cp)*(wcw*cp*(Tcw_in - Tcw_old) + Pch + cs*(1.2*W_nom) ) + Tcw_old
		
		
		# return the states
		return np.array([Tch, Tcw, COP, d, cs, Tch_sp])

	"""
	This function computes the evolution of the system over a DT step, which contains several dt steps
	"""
	def functionF(self,x,u_old,u,t_old,t,simulate):
		# parameters of the function
		cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT = self.pars
		
		x = x.copy()
		
		# number of steps to perform
		Niter = int(DT/dt)
		
		for i in range(Niter):
				U = u_old + i*dt/DT*(u - u_old)
				x = self.functionF_dt(x, U, t_old+i*dt, simulate = simulate)
				
		return x

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
		cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT = self.pars
	
		Tch     = x[0]
		Tcw     = x[1]
		W       = x[4]*W_nom

		# return the output
		return np.array([Tch, Tcw, W])
	
	"""
	This function computes the cooling power of the chiller
	"""
	def functionPch(self,x,u,CS,t,simulate):
		# parameters of the function
		cp, Mch, Mcw, wch, wcw, dTch_nom, dTcw_nom, W_nom, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT = self.pars

		# Output chilled and condenser water
		Tch = x[0]
		Tcw = x[1]
		
		# Input chilled and condenser water 
		Tch_in  = u[0]
		Tcw_in  = u[1]
		
		# Command signal translated into power
		W_in    = CS*W_nom
	
		# nonlinear function, the compressor characteristic
		dW     = np.absolute(W_in)/W_nom
		dCh    = (np.absolute(Tch - Tch_in)-dTch_nom)/dTch_nom
		dCh    = np.max([np.min([dCh, 1]),-1])
		dCw    = (np.absolute(Tcw - Tcw_in)-dTcw_nom)/dTcw_nom
		dCw    = np.max([np.min([dCh, 1]),-1])
		
		
		if simulate :
		
			# the simulated model contains a variable efficiency
			factor = np.max([np.min([1 -0.1*dW**2 -0.2*dCh**2 -0.2*dCw**2 , 1]),0.0])
			
			# simulation of a fault
			if t>=9*3600 and t<=11*3600:
				COP = 2.5
			elif t>=14*3600 and t<=16*3600:
				COP = 3.5
			else:
				COP = 5.0
			COPe = COP*factor	
		else :
			# the model used in the UKF has a fixed idealized value for the COP
			# assuming a nominal condition
			factor = 1
			COP = x[2]
			COPe = COP
		
		# compute the cooling power
		Pch = dW*COP*W_nom*factor

		# return the state
		return np.array([Pch, COPe])

	"""
	get the number of states variables of interest
	"""
	def getNstates(self):
		return self.n_states

	"""
	get the number of ALL states variables
	"""
	def getALLstates(self):
		return self.n_statesTot

	"""
	get the number of states variables
	"""
	def getNoutputs(self):
		return self.n_outputs
