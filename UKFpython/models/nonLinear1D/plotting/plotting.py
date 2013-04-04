import numpy as np
import matplotlib.pyplot as plt
from   pylab import figure

def plotResults(time,stopTime,X,Y,Z,Xhat,Yhat,P,CovZ,Xsmooth,Psmooth):
	# presentation of the results
	fig = plt.figure()
	ax  = fig.add_subplot(211)
	ax.plot(time,X[:,0],'b',label='$x$')
	ax.plot(time,Xhat[:,0],'g',label='$x_{UKF}$')
	ax.fill_between(time, Xhat[:,0]-np.sqrt(P[:,0,0]), Xhat[:,0] +np.sqrt(P[:,0,0]), facecolor='green', interpolate=True, alpha=0.3)
	ax.plot(time,Xsmooth[:,0],'k',label='$x_{SMOOTH}$')
	ax.fill_between(time, Xsmooth[:,0]-np.sqrt(Psmooth[:,0,0]), Xsmooth[:,0] +np.sqrt(Psmooth[:,0,0]), facecolor='black', interpolate=True, alpha=0.3)
	ax.set_xlabel('time steps')
	ax.set_ylabel('state')
	ax.set_xlim([0, stopTime])
	ax.legend()
	ax.grid(True)

	ax  = fig.add_subplot(212)
	ax.plot(time,Y[:,0],'g',label='$y$')
	ax.plot(time,Z[:,0],'r',label='$y_{M}$')
	ax.plot(time,Yhat[:,0],'b',label='$y_{UKF}$')
	ax.fill_between(time, Yhat[:,0]-np.sqrt(CovZ[:,0,0]), Yhat[:,0] +np.sqrt(CovZ[:,0,0]), facecolor='blue', interpolate=True, alpha=0.5)
	ax.set_xlabel('time steps')
	ax.set_ylabel('output')
	ax.set_xlim([0, stopTime])
	ax.legend()
	ax.grid(True)

	

	plt.show()
