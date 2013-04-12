import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure

def plotResults(time,stopTime,X,Y,Z,Xhat,Yhat,P,CovZ,Xsmooth,Psmooth,Xaug,Paug):

	# presentation of the results
	fig = plt.figure()
	ax  = fig.add_subplot(211)
	ax.plot(time,X[:,0],'b',label='$x$')
	ax.plot(time,Xhat[:,0],'r',label='$x_{UKF}$')
	ax.fill_between(time, Xhat[:,0]-np.sqrt(P[:,0,0]), Xhat[:,0] +np.sqrt(P[:,0,0]), facecolor='red', interpolate=True, alpha=0.3)
	ax.plot(time,Xsmooth[:,0],'g',label='$x_{SMOOTH}$')
	ax.fill_between(time, Xsmooth[:,0]-np.sqrt(Psmooth[:,0,0]), Xsmooth[:,0] +np.sqrt(Psmooth[:,0,0]), facecolor='green', interpolate=True, alpha=0.3)
	ax.plot(time,Xaug[:,0],'k',label='$x_{AUG}$')
	ax.fill_between(time, Xaug[:,0]-np.sqrt(Paug[:,0,0]), Xaug[:,0] +np.sqrt(Paug[:,0,0]), facecolor='black', interpolate=True, alpha=0.3)
	ax.set_xlabel('time steps')
	ax.set_ylabel('position')
	ax.set_xlim([0, stopTime])
	ax.legend()
	ax.grid(True)

	ax  = fig.add_subplot(212)
	ax.plot(time,Y[:,0],'b',label='$v$')
	ax.plot(time,Xhat[:,1],'r',label='$v_{UKF}$')
	ax.fill_between(time, Xhat[:,1]-np.sqrt(P[:,1,1]), Xhat[:,1] +np.sqrt(P[:,1,1]), facecolor='red', interpolate=True, alpha=0.3)
	ax.plot(time,Xsmooth[:,1],'g',label='$v_{SMOOTH}$')
	ax.fill_between(time, Xsmooth[:,1]-np.sqrt(Psmooth[:,1,1]), Xsmooth[:,1] +np.sqrt(Psmooth[:,1,1]), facecolor='green', interpolate=True, alpha=0.3)
	ax.plot(time,Xaug[:,1],'k',label='$v_{AUG}$')
	ax.fill_between(time, Xaug[:,1]-np.sqrt(Paug[:,1,1]), Xaug[:,1] +np.sqrt(Paug[:,1,1]), facecolor='black', interpolate=True, alpha=0.3)
	ax.set_xlabel('time steps')
	ax.set_ylabel('velocity')
	ax.set_xlim([0, stopTime])
	ax.legend()
	ax.grid(True)

	# measured outputs
	fig2 = plt.figure()
	ax2  = fig2.add_subplot(211)
	ax2.plot(time,Y[:,1],'g',label='$a$')
	ax2.plot(time,Z[:,1],'ro',label='$a_{M}$')
	ax2.set_xlabel('time steps')
	ax2.set_ylabel('acceleration')
	ax2.set_xlim([0, stopTime])
	ax2.legend()
	ax2.grid(True)

	ax2  = fig2.add_subplot(212)
	ax2.plot(time,Y[:,2],'g',label='$E_{kin}$')
	ax2.plot(time,Z[:,2],'ro',label='$E_{M}$')
	ax2.set_xlabel('time steps')
	ax2.set_ylabel('Kinetic energy')
	ax2.set_xlim([0, stopTime])
	ax2.legend()
	ax2.grid(True)

	plt.show()
