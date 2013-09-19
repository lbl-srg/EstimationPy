'''
Created on Sep 18, 2013

@author: marco
'''
import numpy as np
import matplotlib.pyplot as plt

def plotResults(time,stopTime,U,X,Z,Xhat,Yhat,P,CovZ,Xsmooth,Psmooth,timeSim,Usim,Ysim):
    # presentation of the results
    fig = plt.figure()
    ax  = fig.add_subplot(311)
    ax.plot(time,X,'b',label='$x$')
    ax.plot(time,Xhat[:,0],'r',label='$x_{UKF}$')
    ax.fill_between(time, Xhat[:,0] - np.sqrt(P[:,0,0]), Xhat[:,0] +np.sqrt(P[:,0,0]), facecolor='red', interpolate=True, alpha=0.3)
    ax.plot(time,Xsmooth[:,0],'g',label='$x_{SMOOTH}$')
    ax.fill_between(time, Xsmooth[:,0]-np.sqrt(Psmooth[:,0,0]), Xsmooth[:,0] +np.sqrt(Psmooth[:,0,0]), facecolor='green', interpolate=True, alpha=0.3)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('state')
    ax.set_xlim([0, stopTime])
    ax.legend()
    ax.grid(True)

    ax  = fig.add_subplot(312)
    ax.plot(timeSim,Ysim,'b',label='$y$')
    ax.plot(time,Z,'bo',label='$y_{M}$', alpha=0.3)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('output')
    ax.set_xlim([0, stopTime])
    ax.legend()
    ax.grid(True)
    
    ax  = fig.add_subplot(313)
    ax.plot(timeSim,Usim,'k',label='$u$')
    ax.plot(time,U,'ko',label='$u_{M}$', alpha=0.3)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('input')
    ax.set_xlim([0, stopTime])
    ax.legend()
    ax.grid(True)

    plt.show()
