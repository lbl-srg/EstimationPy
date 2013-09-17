import numpy as np
from ukf.Model import Model

class model(Model):

	# Physical parameters of the chiller
	cp    = 4186.0
	Mch   = 50.0
	Mcw   = 90.0
	W_max = 900000.0
	
	# Nominal conditions
	wch_nom = 91.8
	wcw_nom = 173.5
	Tch_nom = 3.9
	Tcw_nom = 35
	DPch_nom = 26300.0
	DPcw_nom = 39200.0
	
	W_nom    = 800000.0
	CoolCap  = 4000000.0
	COP_nom  = CoolCap/W_nom
	eta_car  = COP_nom*(Tcw_nom - Tch_nom)/(Tch_nom + 273.15)
	
	# Hydraulics parameters
	tau1  = 10
	tau2  = 10
	Pref  = 2e5
	a_V1  = 0.5   # authorities of the valves
	a_V2  = 0.5
	dpV1  = -a_V1/(a_V1-1.0)*DPcw_nom
	dpV2  = -a_V2/(a_V2-1.0)*DPch_nom
	P1ref = Pref + DPcw_nom + dpV1
	P2ref = Pref + DPch_nom + dpV2
	
	Kv1   = wcw_nom/np.sqrt(dpV1)
	Kv2   = wch_nom/np.sqrt(dpV2)
	Kch   = wch_nom/np.sqrt(DPch_nom)
	Kcw   = wcw_nom/np.sqrt(DPcw_nom)
	
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
	pars = (cp, Mch, Mcw, wch_nom, wcw_nom, Tch_nom, Tcw_nom, DPch_nom, DPcw_nom, W_nom, W_max, CoolCap, COP_nom, eta_car, a_V1, a_V2, tau1, tau2, Pref, P1ref, P2ref, dpV1, dpV2, Kv1, Kv2, Kch, Kcw, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT)
	
	# measured outputs are: 
	# Tch: temperature chilled water,
	# Tcw: temperature condenser water,
	# W:   compressor power consumption
	
	# the states of the system are (Tch, Tcw, COP, valve1, valve2, P1, P2, d, CS, Tch_sp)
	# initial state vector is
	X0 = np.array([10.0, 16.0, 0.6, 0, 0, Pref, Pref, 0, 0, 4.0])

	# initialize state and output vectors
	n_states    = 10
	n_statesTot = 10
	n_outputs   = 5

	"""
	Initialize the model with parameters
	"""
	def __init__(self,cp=cp, Mch=Mch, Mcw=Mcw, wch_nom=wch_nom, wcw_nom=wcw_nom, Tch_nom=Tch_nom, Tcw_nom=Tcw_nom, DPch_nom=DPch_nom, DPcw_nom=DPcw_nom, W_nom=W_nom, W_max=W_max, CoolCap=CoolCap, tau1=tau1, tau2=tau2, Pref=Pref, a_V1=a_V1, a_V2=a_V2, Ti=Ti, Td=Td, K=K, N=N, b=b, c=c, CSmax=CSmax, CSmin=CSmin, dt=dt, DT=DT, X0=X0):
		# use the initialization method of inherited class
		Model.__init__(self, 10, 10, 5, X0, None, [1.0, 1.0, 200], [1.0, 1.0, 0.1, 0.0001, 0.0001, 0.0001])
		
		# set parameters of the model (Physical + control system)
		self.setPars(cp, Mch, Mcw, wch_nom, wcw_nom, Tch_nom, Tcw_nom, DPch_nom, DPcw_nom, W_nom, W_max, CoolCap, tau1, tau2, Pref, a_V1, a_V2, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT)
	
	"""
	This method initialize the sampling step
	"""
	def setDT(self,dt=0.1):
		self.dt = dt

	"""
	This method modify the parameter sof the model
	"""
	def setPars(self,cp=cp, Mch=Mch, Mcw=Mcw, wch_nom=wch_nom, wcw_nom=wcw_nom, Tch_nom=Tch_nom, Tcw_nom=Tcw_nom, DPch_nom=DPch_nom, DPcw_nom=DPcw_nom, W_nom=W_nom, W_max=W_max, CoolCap=CoolCap, tau1=tau1, tau2=tau2, Pref=Pref, a_V1=a_V1, a_V2=a_V2, Ti=Ti, Td=Td, K=K, N=N, b=b, c=c, CSmax=CSmax, CSmin=CSmin, dt=dt, DT=DT):
		# parameters of the model
		# Physical parameters of the chiller
		self.cp    = cp
		self.Mch   = Mch
		self.Mcw   = Mcw
		self.W_max = W_max
	
		# Nominal conditions
		self.wch_nom = wch_nom
		self.wcw_nom = wcw_nom
		self.Tch_nom = Tch_nom
		self.Tcw_nom = Tcw_nom
		self.DPch_nom = DPch_nom
		self.DPcw_nom = DPcw_nom
	
		self.W_nom    = W_nom
		self.CoolCap  = CoolCap
		self.COP_nom  = CoolCap/W_nom
		self.eta_car  = self.COP_nom*(Tcw_nom - Tch_nom)/(Tch_nom + 273.15)
	
		# Hydraulics parameters
		self.tau1  = tau1
		self.tau2  = tau2
		self.Pref  = Pref
		self.a_V1  = a_V1   # authorities of the valves
		self.a_V2  = a_V2
		self.dpV1  = -a_V1/(a_V1-1.0)*DPcw_nom
		self.dpV2  = -a_V2/(a_V2-1.0)*DPch_nom
		self.P1ref = Pref + DPcw_nom + self.dpV1
		self.P2ref = Pref + DPch_nom + self.dpV2
	
		self.Kv1   = wcw_nom/np.sqrt(self.dpV1)
		self.Kv2   = wch_nom/np.sqrt(self.dpV2)
		self.Kch   = wch_nom/np.sqrt(DPch_nom)
		self.Kcw   = wcw_nom/np.sqrt(DPcw_nom)
		
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
		self.Tch_nom, self.Tcw_nom, self.DPch_nom, self.DPcw_nom,\
		self.W_nom, self.W_max, self.CoolCap, self.COP_nom, self.eta_car,\
		self.a_V1, self.a_V2, self.tau1, self.tau2, self.Pref, self.P1ref,\
		self.P2ref, self.dpV1, self.dpV2, self.Kv1, self.Kv2, self.Kch, self.Kcw, self.Ti, self.Td, self.K, self.N, self.b,\
		self.c, self.CSmax,self.CSmin, self.dt, self.DT)

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
		cp, Mch, Mcw, wch_nom, wcw_nom, Tch_nom, Tcw_nom, DPch_nom, DPcw_nom, W_nom, W_max, CoolCap, COP_nom, eta_car, a_V1, a_V2, tau1, tau2, Pref, P1ref, P2ref, dpV1, dpV2, Kv1, Kv2, Kch, Kcw, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT = self.pars
		
		# the states of the system are (Tch, Tcw, etaPL, valve1, valve2, P1, P2, d, CS, Tch_sp)
		# state variable of interest
		Tch_old    = x[0]
		Tcw_old    = x[1]
		etaPL_old  = x[2]
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
			if t>=12*3600:
				valve1 = np.min([valve1_old + 0.00001*dt, 0.35])
			else:
				valve1 = 0
			
			if t>=10*3600 and t<=12*3600:
				# valve2 = 0.2
				valve2  = dt/50*( - valve2_old + 0.2 ) + valve2_old
			else:
				# valve2 = 0
				valve2  = dt/50*( - valve2_old + 0.0 ) + valve2_old
				
		
		else:
			valve1 = valve1_old
			valve2 = valve2_old
		
		
		# Get the cooling POwer and the COP
		(Pch, etaPL) = self.functionPch(x,u,cs,t,simulate)
		
		# Chilled and Condenser water flows
		try:
			K1  = np.sqrt( 1/( 1/(Kv2**2)/(((1-valve1)**2)*(self.fvalve(CMD_V1))**2) + 1/(Kcw**2) ))
			wcw = K1*self.sqrtReg(P1_old - Pref)
		except ValueError:
			print "At time "+str(t)
			print "P1_old - Pref = "+str(P1_old - Pref)
		
		try:
			K2  = np.sqrt( 1/( 1/(Kv1**2)/(((1-valve2)**2)*(self.fvalve(CMD_V2))**2) + 1/(Kch**2) ))
			wch = K2*self.sqrtReg(P2_old - Pref)
		except ValueError:
			print "At time "+str(t)
			print "P2_old - Pref = "+str(P2_old - Pref)

		# new state values (Chiller)
		Tch = dt/(Mch*cp)*(wch*cp*(Tch_in - Tch_old) - Pch ) + Tch_old
		Tcw = dt/(Mcw*cp)*(wcw*cp*(Tcw_in - Tcw_old) + Pch + cs*W_max ) + Tcw_old
		
		# new state values (Pumps)
		P1  = dt/tau1*( - P1_old + (DPcw_nom + dpV1)*CMD_P1 + Pref ) + P1_old
		P2  = dt/tau2*( - P2_old + (DPch_nom + dpV2)*CMD_P2 + Pref ) + P2_old		
		
		# return the states
		# (Tch, Tcw, COP, valve1, valve2, P1, P2, d, CS, Tch_sp)
		if simulate:
			return np.array([Tch, Tcw, etaPL, valve1, valve2, P1, P2, d, cs, Tch_sp])
		else:
			return np.array([Tch, Tcw, etaPL, valve1, valve2, P1, P2])	
				
	"""
	This function computes the evolution of the system over a DT step, which contains several dt steps
	"""
	def functionF(self, val):
		(x,u_old,u,t_old,t,simulate) = val
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
			K2  = np.sqrt( 1/( 1/(self.Kv1**2)/((self.fvalve(CMD_V2, occ=valve2))**2) + 1/(self.Kch**2) ))
			wch = K2*self.sqrtReg(P2_old - self.Pref)
		except ValueError:
			print "At time "+str(t)
			print "P2_old - Pref = "+str(P2_old - self.Pref)
			
			
		# The power consumed, output of the simulation model
		if simulate:
			W   = x[8]*self.W_max
			
			# return the output
			return np.array([Tch, Tcw, wch, W, self.etaPL(x[8])])
		else:
			W   = u[2]*self.W_max
			
			# return the output
			return np.array([Tch, Tcw, wch])
			
	"""
	This function computes the cooling power of the chiller
	"""
	def functionPch(self,x,u,CS,t,simulate):
		# parameters of the function
		cp, Mch, Mcw, wch_nom, wcw_nom, Tch_nom, Tcw_nom, DPch_nom, DPcw_nom, W_nom, W_max, CoolCap, COP_nom, eta_car, a_V1, a_V2, tau1, tau2, Pref, P1ref, P2ref, dpV1, dpV2, Kv1, Kv2, Kch, Kcw, Ti, Td, K, N, b, c, CSmax, CSmin, dt, DT = self.pars
		
		# Output chilled and condenser water
		Tch = x[0]
		Tcw = x[1]
		
		# Input chilled and condenser water 
		Tch_in  = u[0]
		Tcw_in  = u[1]
		
		# power of teh compressor
		P = CS*W_max
		# Carnot COP
		COP_car = (Tch + 273.15)/(Tcw - Tch)
		
		
		if simulate :
			
			# simulation of a fault introducing a scaling factor that reduces etaPL
			if t>=9*3600 and t<=9.5*3600:
				factor = (1 - (t-9*3600)*(0.65)/1800)
			elif t>=9.5*3600 and t<=10.5*3600:
				factor = 0.35
			elif t>=10.5*3600 and t<=11*3600:
				factor = 0.35 + (t-10.5*3600)*(0.65)/1800	
			elif t>=14*3600 and t<=14.3*3600:
				factor = (1 - (t-14*3600)*(0.4)/(0.3*3600))
			elif t>=14.3*3600 and t<=16*3600:
				factor = 0.6
			elif t>=16*3600 and t<=16.3*3600:
				factor = 0.6 + (t-16*3600)*(0.4)/(3600*0.3)
			else:
				factor = 1.0
			
			etaPL   = self.etaPL(CS)*factor
		else :
			# the model used in the UKF has a fixed idealized value for the etaPL
			# assuming a nominal condition
			etaPL = x[2]	
		
		# Overall COP
		COP     = eta_car*COP_car*etaPL
		
		# compute the cooling power
		Pch = COP*P

		# return the state
		return np.array([Pch, etaPL])

	"""
	Valve characteristics
	
	occ: occluded portion of the valve
	ope: open portion of the valve
	"""
	def fvalve(self,cmd,occ=0.0,ope=0.0):
	
		occ = np.min([1.0, np.max([0.0, occ])])
	
		cmdFault = np.min([1.0-occ, np.max([ope, cmd])])
	
		x = np.min([1.0, np.max([0.0, self.sqrtReg(cmdFault)]) ])
	
		return x
	
	"""
	Regularized sqrt root function
	"""
	def sqrtReg(self,x,delta=0.01):
		return x/np.sqrt(np.sqrt(x**2.0 + delta**2.0))
	
	"""
	Efficiency of the compressor and cycle as function of the CS [0,1]
	"""
	def etaPL(self,CS):
		x_nom   = float(self.W_nom/self.W_max)
		min_eta = 0.6
		a       = (min_eta - 1)/(x_nom**2)
		b       = -(2*min_eta -2)/x_nom
		c       = min_eta
		y       = a*CS**2 + b*CS + c
		return y  
	
	def plotEtaPL(self):
		import matplotlib.pyplot as plt
		
		x   = np.linspace(0.0, 1.0, 100)
		eta = self.etaPL(x)
		etaMin = eta - 0.1*np.ones(eta.shape)
		etaMax = np.zeros(eta.shape)
		i = 0
		for e in eta:
			etaMax[i] = np.min([e + 0.1, 1.05])
			i        += 1
		
		fig = plt.figure()
		fig.set_size_inches(10,5)
		ax  = fig.add_subplot(111)
		ax.plot(x,eta,'b',label='$\eta_{PL}$')
		ax.fill_between(x, etaMin, etaMax, facecolor='blue', interpolate=True, alpha=0.3)
		ax.set_ylabel('Efficiency [$\cdot$]')
		ax.set_xlabel('Control Signal [$\cdot$]')
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.4, 1.1])
		legend = ax.legend(loc='upper left',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
		legend.draggable()
		ax.grid(True)
		plt.savefig('eta.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
		return