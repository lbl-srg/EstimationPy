import numpy as np

class model():

	# Physical parameters of the chiller
	cp  = 4186
	Mch = 50
	Mcw = 90
	
	# Nominal conditions
	wch_nom = 91.8
	wcw_nom = 173.5
	dTch_nom = 8.9
	dTcw_nom = 5.6
	DPch_nom = 26300
	DPcw_nom = 39200
	W_nom    = 800000
	W_max    = 900000
	CoolCap  = 4000000
	COP_nom  = CoolCap/W_nom
	
	# Efficiency of the chiller
	alpha = 0.075
	beta  = 0.1
	gamma = 0.1
	
	# Hydraulics parameters
	tau1  = 10
	tau2  = 10
	Pref  = 200000
	P1ref = Pref + DPcw_nom
	P2ref = Pref + DPch_nom
	CV1   = wcw_nom/np.sqrt(DPcw_nom)
	CV2   = wch_nom/np.sqrt(DPch_nom)
	
	# Controller parameters
	Ti = 10
	Td = 5
	K  = -10
	CSmax = 1
	CSmin = 0
	b     = 1
	c     = 1
	N     = 8
	
	# step of the time discretization: dt
	# step of the sampling: DT
	# DT >= dt
	dt  = 0.1
	DT  = 60.0
	
	# parameters
	pars = (cp, Mch, Mcw, wch_nom, wcw_nom, dTch_nom, dTcw_nom, DPch_nom, DPcw_nom, W_nom, W_max, CoolCap, COP_nom, alpha, beta, gamma, tau1, tau2, Pref, P1ref, P2ref, CV1, CV2, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT)
	
	# measured outputs are: 
	# Tch: temperature chilled water,
	# Tcw: temperature condenser water,
	# W:   compressor power consumption
	
	# the states of the system are (Tch, Tcw, COP, valve1, valve2, P1, P2, d, CS, Tch_sp)
	# initial state vector is
	X0 = np.array([10.0, 16.0, COP_nom, 0, 0, Pref, Pref, 0, 0, 4.0])

	# initialize state and output vectors
	n_states    = 10
	n_statesTot = 10
	n_outputs   = 4

	"""
	Initialize the model with parameters
	"""
	def __init__(self,cp=cp, Mch=Mch, Mcw=Mcw, wch_nom=wch_nom, wcw_nom=wcw_nom, dTch_nom=dTch_nom, dTcw_nom=dTcw_nom, DPch_nom=DPch_nom, DPcw_nom=DPcw_nom, W_nom=W_nom, W_max=W_max, CoolCap=CoolCap, COP_nom=COP_nom, alpha=alpha, beta=beta, gamma=gamma, tau1=tau1, tau2=tau2, Pref=Pref, P1ref=P1ref, P2ref=P2ref, CV1=CV1, CV2=CV2, Ti=Ti, Td=Td, K=K, N=N, b=b, c=c, CSmax=CSmax, CSmin=CSmin, dt=dt, DT=DT, X0=X0):
		# parameters of the model
		self.cp  = cp
		self.Mch = Mch
		self.Mcw = Mcw
		
		self.alpha = alpha
		self.beta  = beta
		self.gamma = gamma
		
		self.wch_nom = wch_nom
		self.wcw_nom = wcw_nom
		self.dTch_nom = dTch_nom
		self.dTcw_nom = dTcw_nom
		self.DPch_nom = DPch_nom
		self.DPcw_nom = DPcw_nom
		self.W_nom    = W_nom
		self.W_max    = W_max
		self.CoolCap  = CoolCap
		self.COP_nom  = CoolCap/W_nom
		
		
		self.tau1  = tau1
		self.tau2  = tau2
		self.Pref  = Pref
		self.P1ref = Pref + DPcw_nom
		self.P2ref = Pref + DPch_nom
		self.CV1   = wcw_nom/np.sqrt(DPcw_nom)
		self.CV2   = wch_nom/np.sqrt(DPch_nom)
		
		self.Ti    = Ti
		self.Td    = Td
		self.K     = K
		self.CSmax = CSmax
		self.CSmin = CSmin
		self.b     = b
		self.c     = c
		self.N     = N
		
		self.dt  = dt
		self.DT  = DT
		
		self.pars = \
		(self.cp, self.Mch, self.Mcw, self.wch_nom, self.wcw_nom,\
		self.dTch_nom, self.dTcw_nom, self.DPch_nom, self.DPcw_nom,\
		self.W_nom, self.W_max, self.CoolCap, self.COP_nom, self.alpha,\
		self.beta, self.gamma, self.tau1, self.tau2, self.Pref, self.P1ref,\
		self.P2ref, self.CV1,self.CV2, self.Ti, self.Td, self.K, self.N, self.b,\
		self.c, self.CSmax,self.CSmin, self.dt, self.DT)

		# measurement covariance noise
		R     = np.diag([1.0, 1.0, 200])
		sqrtR = np.linalg.cholesky(R)
		 
		# initial state vector (position, velocity)
		self.X0 = X0

		# process noise
		Q     = np.diag([1.0, 1.0, 0.1, 0.0001, 0.0001, 0.0001])
		sqrtQ = np.linalg.cholesky(Q)
	
		# initialize state and output vectors
		n_states    = 10
		n_statesTot = 10
		n_outputs   = 3

	"""
	This method define the initial state
	"""
	def setInitialState(self,X0):
		self.X0 = X0	

	"""
	This method returns the initial state
	"""
	def getInitialState(self):
		return self.X0	
	
	"""
	This method initialize the sampling step
	"""
	def setDT(self,dt=0.1):
		self.dt = dt

	"""
	This method modify the parameter sof the model
	"""
	def setPars(self,cp=cp, Mch=Mch, Mcw=Mcw, wch_nom=wch_nom, wcw_nom=wcw_nom, dTch_nom=dTch_nom, dTcw_nom=dTcw_nom, DPch_nom=DPch_nom, DPcw_nom=DPcw_nom, W_nom=W_nom, W_max=W_max, CoolCap=CoolCap, COP_nom=COP_nom, alpha=alpha, beta=beta, gamma=gamma, tau1=tau1, tau2=tau2, Pref=Pref, P1ref=P1ref, P2ref=P2ref, CV1=CV1, CV2=CV2, Ti=Ti, Td=Td, K=K, N=N, b=b, c=c, CSmax=CSmax, CSmin=CSmin, dt=dt, DT=DT):
		# parameters of the model
		self.cp  = cp
		self.Mch = Mch
		self.Mcw = Mcw
		
		self.alpha = alpha
		self.beta  = beta
		self.gamma = gamma
		
		self.wch_nom = wch_nom
		self.wcw_nom = wcw_nom
		self.dTch_nom = dTch_nom
		self.dTcw_nom = dTcw_nom
		self.DPch_nom = DPch_nom
		self.DPcw_nom = DPcw_nom
		self.W_nom    = W_nom
		self.W_max    = W_max
		self.CoolCap  = CoolCap
		self.COP_nom  = CoolCap/W_nom
		
		
		self.tau1  = tau1
		self.tau2  = tau2
		self.Pref  = Pref
		self.P1ref = Pref + DPcw_nom
		self.P2ref = Pref + DPch_nom
		self.CV1   = wcd_nom/np.sqrt(DPcw_nom)
		self.CV2   = wch_nom/np.sqrt(DPch_nom)
		
		self.Ti    = Ti
		self.Td    = Td
		self.K     = K
		self.CSmax = CSmax
		self.CSmin = CSmin
		self.b     = b
		self.c     = c
		self.N     = N
		
		self.dt  = dt
		self.DT  = DT
		
		self.pars = \
		(self.cp, self.Mch, self.Mcw, self.wch_nom, self.wcw_nom,\
		self.dTch_nom, self.dTcw_nom, self.DPch_nom, self.DPcw_nom,\
		self.W_nom, self.W_max, self.CoolCap, self.COP_nom, self.alpha,\
		self.beta, self.gamma, self.tau1, self.tau2, self.Pref, self.P1ref,\
		self.P2ref, self.CV1,self.CV2, self.Ti, self.Td, self.K, self.N, self.b,\
		self.c, self.CSmax,self.CSmin, self.dt, self.DT)

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
		cp, Mch, Mcw, wch_nom, wcw_nom, dTch_nom, dTcw_nom, DPch_nom, DPcw_nom, W_nom, W_max, CoolCap, COP_nom, alpha, beta, gamma, tau1, tau2, Pref, P1ref, P2ref, CV1, CV2, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT = self.pars
		
		# the states of the system are (Tch, Tcw, COP, valve1, valve2, P1, P2, d, CS, Tch_sp)
		# state variable of interest
		Tch_old    = x[0]
		Tcw_old    = x[1]
		COP_old    = x[2]
		valve1_old = x[3]
		valve2_old = x[4]
		P1_old     = x[5]
		P2_old     = x[6]
		
		# input variables
		Tch_in  = u[0]
		Tcw_in  = u[1]
		
		if simulate:
			Tch_sp  = u[2]
		else:
			cs      = u[2]
			
		CMD_P1  = u[3]
		CMD_P2  = u[4]
		CMD_V1  = u[5]
		CMD_V2  = u[6]
		
		# During the simulation the control action is computed
		if simulate:			
			# state variables for the controller
			D_old   = x[7]
			CS_old  = x[8]
			SP_old  = x[9]
			PV_old  = Tch_old
		
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
			
			# simulate a fault in the valve
			if t>=9*3600:
				valve1 = np.min([valve1_old + 0.00001*dt, 0.35])
			else:
				valve1 = 0
			
			if t>=10*3600 and t<=12*3600:
				valve2 = 0.15
			else:
				valve2 = 0
		
		else:
			valve1 = valve1_old
			valve2 = valve2_old
		
		
		# Get the cooling POwer and the COP
		(Pch, COP) = self.functionPch(x,u,cs,t,simulate)
		
		# Chilled and Condenser water flows
		try:
			wcw = CV1*(1-valve1)*self.fvalve(CMD_V1)*np.sqrt(P1_old - Pref)
		except ValueError:
			print "At time "+str(t)
			print "P1_old - Pref = "+str(P1_old - Pref)
		
		try:
			wch = CV2*(1-valve2)*self.fvalve(CMD_V2)*np.sqrt(P2_old - Pref)
		except ValueError:
			print "At time "+str(t)
			print "P2_old - Pref = "+str(P2_old - Pref)

		# new state values (Chiller)
		Tch = dt/(Mch*cp)*(wch*cp*(Tch_in - Tch_old) - Pch ) + Tch_old
		Tcw = dt/(Mcw*cp)*(wcw*cp*(Tcw_in - Tcw_old) + Pch + cs*W_max ) + Tcw_old
		
		# new state values (Pumps)
		P1  = dt/tau1*( - P1_old + DPcw_nom*CMD_P1 + Pref ) + P1_old
		P2  = dt/tau2*( - P2_old + DPch_nom*CMD_P2 + Pref ) + P2_old		
		
		# return the states
		# (Tch, Tcw, COP, valve1, valve2, P1, P2, d, CS, Tch_sp)
		if simulate:
			return np.array([Tch, Tcw, COP, valve1, valve2, P1, P2, d, cs, Tch_sp])
		else:
			return np.array([Tch, Tcw, COP, valve1, valve2, P1, P2])	
				
	"""
	This function computes the evolution of the system over a DT step, which contains several dt steps
	"""
	def functionF(self,x,u_old,u,t_old,t,simulate):
		x = x.copy()
		
		# number of steps to perform
		Niter = int(self.DT/self.dt)
		
		for i in range(Niter):
				U = u_old + i*self.dt/self.DT*(u - u_old)
				x = self.functionF_dt(x, U, t_old + i*self.dt, simulate = simulate)
				
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
		
		# output variables of the model
		Tch     = x[0]
		Tcw     = x[1]
		
		# the chilled water mass flow rate
		valve2  = x[4]
		P2_old  = x[6]
		CMD_V2  = u[6]
		
		try:
			wch = self.CV2*(1-valve2)*self.fvalve(CMD_V2)*np.sqrt(P2_old - self.Pref)
		except ValueError:
			print "At time "+str(t)
			print "P2_old - Pref = "+str(P2_old - self.Pref)
			
			
		# The power consumed, output of the simulation model
		if simulate:
			W   = x[8]*self.W_max
			
			# return the output
			return np.array([Tch, Tcw, wch, W])
		else:
			W   = u[2]*self.W_max
			
			# return the output
			return np.array([Tch, Tcw, wch])
			
	"""
	This function computes the cooling power of the chiller
	"""
	def functionPch(self,x,u,CS,t,simulate):
		# parameters of the function
		cp, Mch, Mcw, wch_nom, wcw_nom, dTch_nom, dTcw_nom, DPch_nom, DPcw_nom, W_nom, W_max, CoolCap, COP_nom, alpha, beta, gamma, tau1, tau2, Pref, P1ref, P2ref, CV1, CV2, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT = self.pars

		# Output chilled and condenser water
		Tch = x[0]
		Tcw = x[1]
		
		# Input chilled and condenser water 
		Tch_in  = u[0]
		Tcw_in  = u[1]
		
		# Command signal translated into power
		W_in    = CS*W_max
	
		# nonlinear function, the compressor characteristic
		dW     = np.absolute(W_nom - W_in)/W_nom
		dCh    = (np.absolute(Tch - Tch_in)-dTch_nom)/dTch_nom
		dCh    = np.max([np.min([dCh, 1]),-1])
		dCw    = (np.absolute(Tcw - Tcw_in)-dTcw_nom)/dTcw_nom
		dCw    = np.max([np.min([dCh, 1]),-1])
		
		if simulate :
		
			# the simulated model contains a variable efficiency
			factor = np.max([np.min([1 -alpha*dW**2 -beta*dCh**2 -gamma*dCw**2 , 1]),0.0])
			
			# simulation of a fault
			if t>=9*3600 and t<=11*3600:
				COP = COP_nom*0.35
			elif t>=14*3600 and t<=16*3600:
				COP = COP_nom*0.6
			else:
				COP = COP_nom
			
			COPe = COP*factor	
		else :
			# the model used in the UKF has a fixed idealized value for the COP
			# assuming a nominal condition
			factor = 1
			COP = x[2]
			COPe = COP
		
		# compute the cooling power
		Pch = W_in*COP*factor

		# return the state
		return np.array([Pch, COPe])

	"""
	Valve characteristics
	"""
	def fvalve(self,cmd):
		x = np.min([1.0, np.max([0.0, cmd]) ])
		return x
	
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
