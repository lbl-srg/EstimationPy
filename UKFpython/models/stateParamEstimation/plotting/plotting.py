import numpy as np
import matplotlib.pyplot as plt
from  pylab import figure
import numpy as np

def plotResults(time,stopTime,X,Y,Z,U,Um,Xhat,Yhat,P,covY,Xsmooth,Psmooth,Xaug,Paug):
		
	# plotting input and oputputs (real and measured)
	fig = plt.figure()
	ax1  = fig.add_subplot(211)
	ax1.plot(time,U[:,0],'b',label='$u$')
	ax1.plot(time,Um[:,0],'bx',label='$u_{m}$')
	ax1.set_xlabel('time [s]')
	ax1.set_ylabel('input')
	ax1.set_xlim([0, stopTime])
	ax1.legend()
	ax1.grid(True)

	ax1  = fig.add_subplot(212)
	ax1.plot(time,Y[:,0],'g',label='$y$')
	ax1.plot(time,Z[:,0],'gx',label='$y_{m}$')
	ax1.set_xlabel('time [s]')
	ax1.set_ylabel('output')
	ax1.set_xlim([0, stopTime])
	ax1.legend()
	ax1.grid(True)

	# plotting real state and its estimation
	fig2 = plt.figure()
	ax2  = fig2.add_subplot(311)
	ax2.plot(time,X[:,0],'b',label='$x_{real}$')
	ax2.plot(time,Xhat[:,0],'r--',label='$x_{UKF}$')
	ax2.plot(time,Xsmooth[:,0],'g--',label='$x_{SMOOTH}$')
	ax2.plot(time,Xaug[:,0],'k--',label='$x_{AUG}$')
	ax2.fill_between(time, Xhat[:,0]-2*np.sqrt(P[:,0,0]), Xhat[:,0] +2*np.sqrt(P[:,0,0]), facecolor='red', interpolate=True, alpha=0.3)
	ax2.fill_between(time, Xsmooth[:,0]-2*np.sqrt(Psmooth[:,0,0]), Xsmooth[:,0] +2*np.sqrt(Psmooth[:,0,0]), facecolor='green', interpolate=True, alpha=0.3)
	ax2.fill_between(time, Xaug[:,0]-2*np.sqrt(Paug[:,0,0]), Xaug[:,0] +2*np.sqrt(Paug[:,0,0]), facecolor='black', interpolate=True, alpha=0.3)
	ax2.set_xlabel('time [s]')
	ax2.set_ylabel('state')
	ax2.set_xlim([0, stopTime])
	ax2.legend()
	ax2.grid(True)

	ax2  = fig2.add_subplot(312)
	ax2.plot(time,X[:,1],'b',label='$b_{real}$')
	ax2.plot(time,Xhat[:,1],'r--',label='$b_{UKF}$')
	ax2.plot(time,Xsmooth[:,1],'g--',label='$b_{SMOOTH}$')
	ax2.plot(time,Xaug[:,1],'k--',label='$b_{AUG}$')
	ax2.fill_between(time, Xhat[:,1]-2*np.sqrt(P[:,1,1]), Xhat[:,1] +2*np.sqrt(P[:,1,1]), facecolor='red', interpolate=True, alpha=0.3)
	ax2.fill_between(time, Xsmooth[:,1]-2*np.sqrt(Psmooth[:,1,1]), Xsmooth[:,1] +2*np.sqrt(Psmooth[:,1,1]), facecolor='green', interpolate=True, alpha=0.3)
	ax2.fill_between(time, Xaug[:,1]-2*np.sqrt(Paug[:,1,1]), Xaug[:,1] +2*np.sqrt(Paug[:,1,1]), facecolor='black', interpolate=True, alpha=0.3)
	ax2.set_xlabel('time [s]')
	ax2.set_ylabel('parameter')
	ax2.set_xlim([0, stopTime])
	ax2.legend()
	ax2.grid(True)
	
	ax2  = fig2.add_subplot(313)
	ax2.plot(time, np.sqrt(Paug[:,0,0]),'b',label='$\sigma_{x}$')
	ax2.plot(time, np.sqrt(Paug[:,1,1]),'r',label='$\sigma_{b}$')
	ax2.set_xlabel('time [s]')
	ax2.set_ylabel('Covariance')
	ax2.set_xlim([0, stopTime])
	ax2.legend()
	ax2.grid(True)

	plt.show()
