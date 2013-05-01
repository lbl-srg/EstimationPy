import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure

def plotBasic(time,timeSamples,startTime,stopTime,X,Y,U,Um,Z,Xhat,P,Yhat,CovY,Xsmooth,Psmooth):

	#######################################################################
	# TRUE SYSTEM
	fig2 = plt.figure()
	ax2  = fig2.add_subplot(411)
	ax2.plot(1.0/3600.0*time,U[:,0],'b',label='$T_{ch IN}$')
	ax2.plot(1.0/3600.0*timeSamples,Um[:,0],'bo',label='$T_{ch InS}$',alpha=0.2)
	ax2.plot(1.0/3600.0*time,U[:,1],'r',label='$T_{cd IN}$')
	ax2.plot(1.0/3600.0*timeSamples,Um[:,1],'ro',label='$T_{cd InS}$',alpha=0.2)
	ax2.plot(1.0/3600.0*time,U[:,2],'g',label='$T_{ch SP}$')
	ax2.plot(1.0/3600.0*timeSamples,Um[:,2],'go',label='$T_{cd InS}$',alpha=0.2)
	ax2.set_xlabel('time [h]')
	ax2.set_ylabel('Input flows')
	ax2.set_xlim([startTime/3600.0, stopTime/3600.0])
	ax2.legend()
	ax2.grid(True)

	
	ax2  = fig2.add_subplot(412)
	ax2.plot(1.0/3600.0*time,U[:,2],'g',label='$SP$')
	ax2.plot(1.0/3600.0*time,Y[:,0],'b',label='$Tch$')
	ax2.plot(1.0/3600.0*timeSamples,Z[:,0],'bo',label='$Tch S$',alpha=0.2)
	ax2.plot(1.0/3600.0*time,Y[:,1],'r',label='$Tcd$')
	ax2.plot(1.0/3600.0*timeSamples,Z[:,1],'ro',label='$Tcd S$',alpha=0.2)
	ax2.set_xlabel('time [h]')
	ax2.set_ylabel('Controlled variables')
	ax2.set_xlim([startTime/3600.0, stopTime/3600.0])
	ax2.legend()
	ax2.grid(True)
	
	ax2  = fig2.add_subplot(413)
	ax2.plot(1.0/3600.0*time,Y[:,3],'g',label='$W$')
	ax2.plot(1.0/3600.0*timeSamples,Z[:,3],'go',label='$W S$')
	ax2.set_xlabel('time [h]')
	ax2.set_ylabel('Power')
	ax2.set_xlim([startTime/3600.0, stopTime/3600.0])
	ax2.legend()
	ax2.grid(True)
	
	ax2  = fig2.add_subplot(414)
	ax2.plot(1.0/3600.0*time,Y[:,2],'k',label='$\dot{m}_{CH}$')
	ax2.plot(1.0/3600.0*timeSamples,Z[:,2],'k--',label='$\dot{m}_{CH} S$')
	ax2.set_xlabel('time [h]')
	ax2.set_ylabel('CHW flow')
	ax2.set_xlim([startTime/3600.0, stopTime/3600.0])
	ax2.legend()
	ax2.grid(True)
	
	#######################################################################
	fig = plt.figure()
	ax  = fig.add_subplot(311)
	ax.plot(1.0/3600.0*time,X[:,2],'b',label='$COP$')
	ax.plot(1.0/3600.0*timeSamples,Xhat[:,2],'r',label='$COP_{UKF}$')
	ax.fill_between(1.0/3600.0*timeSamples, Xhat[:,2] - np.sqrt(P[:,2,2]), Xhat[:,2] + np.sqrt(P[:,2,2]), facecolor='red', interpolate=True, alpha=0.3)
	ax.plot(1.0/3600.0*timeSamples,Xsmooth[:,2],'g--',label='$COP_{SMOOTH}$')
	ax.fill_between(1.0/3600.0*timeSamples, Xsmooth[:,2] - np.sqrt(Psmooth[:,2,2]), Xsmooth[:,2] + np.sqrt(Psmooth[:,2,2]), facecolor='green', interpolate=True, alpha=0.3)
	ax.set_xlabel('time [h]')
	ax.set_ylabel('COP')
	ax.set_xlim([startTime/3600.0, stopTime/3600.0])
	ax.legend()
	ax.grid(True)
	
	ax  = fig.add_subplot(312)
	ax.plot(1.0/3600.0*time,X[:,3],'b',label='$V_1$')
	ax.plot(1.0/3600.0*timeSamples,Xhat[:,3],'b--',label='$V1_{UKF}$')
	ax.fill_between(1.0/3600.0*timeSamples, Xhat[:,3] - np.sqrt(P[:,3,3]), Xhat[:,3] + np.sqrt(P[:,3,3]), facecolor='blue', interpolate=True, alpha=0.3)
	ax.plot(1.0/3600.0*timeSamples,Xsmooth[:,3],'g--',label='$V1_{SMOOTH}$')
	ax.fill_between(1.0/3600.0*timeSamples, Xsmooth[:,3] - np.sqrt(Psmooth[:,3,3]), Xsmooth[:,3] + np.sqrt(Psmooth[:,3,3]), facecolor='green', interpolate=True, alpha=0.3)
	ax.set_xlabel('time [h]')
	ax.set_ylabel('$valve_1$')
	ax.set_xlim([startTime/3600.0, stopTime/3600.0])
	ax.legend()
	ax.grid(True)
	
	ax  = fig.add_subplot(313)
	ax.plot(1.0/3600.0*time,X[:,4],'r',label='$V_2$')
	ax.plot(1.0/3600.0*timeSamples,Xhat[:,4],'r--',label='$V2_{UKF}$')
	ax.fill_between(1.0/3600.0*timeSamples, Xhat[:,4] - np.sqrt(P[:,4,4]), Xhat[:,4] + np.sqrt(P[:,4,4]), facecolor='red', interpolate=True, alpha=0.3)
	ax.plot(1.0/3600.0*timeSamples,Xsmooth[:,4],'g--',label='$V2_{SMOOTH}$')
	ax.fill_between(1.0/3600.0*timeSamples, Xsmooth[:,4] - np.sqrt(Psmooth[:,4,4]), Xsmooth[:,4] + np.sqrt(Psmooth[:,4,4]), facecolor='green', interpolate=True, alpha=0.3)
	ax.set_xlabel('time [h]')
	ax.set_ylabel('$valve_2$')
	ax.set_xlim([startTime/3600.0, stopTime/3600.0])
	ax.legend()
	ax.grid(True)
	
	#######################################################################
	fig = plt.figure()
	ax3  = fig.add_subplot(111)
	ax3.plot(1.0/3600.0*time,X[:,0],'b',label='$Tch$')
	ax3.plot(1.0/3600.0*timeSamples,Xhat[:,0],'b--',label='$Tch_{UKF}$')
	ax3.fill_between(1.0/3600.0*timeSamples, Xhat[:,0] - np.sqrt(P[:,0,0]), Xhat[:,0] + np.sqrt(P[:,0,0]), facecolor='blue', interpolate=True, alpha=0.3)
	ax3.plot(1.0/3600.0*timeSamples,Xsmooth[:,0],'g--',label='$Tch_{SMOOTH}$')
	ax3.fill_between(1.0/3600.0*timeSamples, Xsmooth[:,0] - np.sqrt(Psmooth[:,0,0]), Xsmooth[:,0] + np.sqrt(Psmooth[:,0,0]), facecolor='green', interpolate=True, alpha=0.3)
	ax3.plot(1.0/3600.0*time,U[:,2],'k',label='$T_{ch SP}$')
	
	ax3.plot(1.0/3600.0*time,X[:,1],'r',label='$Tcd$')
	ax3.plot(1.0/3600.0*timeSamples,Xhat[:,1],'r--',label='$Tcd_{UKF}$')
	ax3.fill_between(1.0/3600.0*timeSamples, Xhat[:,1] - np.sqrt(P[:,1,1]), Xhat[:,1] +np.sqrt(P[:,1,1]), facecolor='red', interpolate=True, alpha=0.3)
	ax3.plot(1.0/3600.0*timeSamples,Xsmooth[:,1],'g--',label='$Tcd_{SMOOTH}$')
	ax3.fill_between(1.0/3600.0*timeSamples, Xsmooth[:,1] - np.sqrt(Psmooth[:,1,1]), Xsmooth[:,1] + np.sqrt(Psmooth[:,1,1]), facecolor='green', interpolate=True, alpha=0.3)
	
	ax3.set_xlabel('time [h]')
	ax3.set_ylabel('Out Temperatures')
	ax3.set_xlim([startTime/3600.0, stopTime/3600.0])
	ax3.legend()
	ax3.grid(True)
	
	plt.show()
	
	

def plotResults(time,stopTime,X,Y,U,Um,Pch,Z,Xhat,Yhat,P,CovZ,Xsmooth,Psmooth,Xaug,Paug):

	return
