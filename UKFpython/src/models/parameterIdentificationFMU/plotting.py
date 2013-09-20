'''
Created on Sep 18, 2013

@author: marco
'''
import numpy as np
import matplotlib.pyplot as plt

def plotResults(pars, truePars):
    # presentation of the results
    fig = plt.figure()
    
    N, C = np.shape(pars)
    a = truePars[0]*np.ones(N)
    b = truePars[1]*np.ones(N)
    c = truePars[2]*np.ones(N)
    d = truePars[3]*np.ones(N)
    
    ax  = fig.add_subplot(111)
    ax.plot(a,'b',label='$a$')
    ax.plot(pars[:,0],'b--',label='$\hat{a}$', alpha=0.8)
    ax.plot(b,'r',label='$b$')
    ax.plot(pars[:,1],'r--',label='$\hat{b}$', alpha=0.8)
    ax.plot(c,'g',label='$c$')
    ax.plot(pars[:,2],'g--',label='$\hat{c}$', alpha=0.8)
    ax.plot(d,'k',label='$d$')
    ax.plot(pars[:,3],'k--',label='$\hat{d}$', alpha=0.8)
    
    ax.set_xlabel('iterations')
    ax.set_ylabel('parameters')
    ax.legend()
    ax.grid(True)
    
    plt.show()

def intermediatePlot(time, X, Z, Xhat, S, Yhat, Sy, Xsmooth, Ssmooth):
    
    # presentation of the results
    fig = plt.figure()
    ax  = fig.add_subplot(211)
    ax.plot(time,X,'b',label='$x$')
    ax.plot(time,Xhat[:,0],'r',label='$x_{UKF}$')
    ax.fill_between(time, Xhat[:,0] - S[:,0,0], Xhat[:,0] + S[:,0,0], facecolor='red', interpolate=True, alpha=0.3)
    ax.plot(time,Xsmooth[:,0],'g',label='$x_{SMOOTH}$')
    ax.fill_between(time, Xsmooth[:,0] - Ssmooth[:,0,0], Xsmooth[:,0] + Ssmooth[:,0,0], facecolor='green', interpolate=True, alpha=0.3)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('state')
    ax.set_xlim([0, time[-1]])
    ax.legend()
    ax.grid(True)

    ax  = fig.add_subplot(212)
    ax.plot(time,Yhat,'b',label='$y$')
    ax.plot(time,Z,'bo',label='$y_{M}$', alpha=0.3)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('output')
    ax.set_xlim([0, time[-1]])
    ax.legend()
    ax.grid(True)
    
    fig2 = plt.figure()
    ax2  = fig2.add_subplot(111)
    ax2.plot(time,Xsmooth[:,1],'b',label='$a$')
    ax2.plot(time,Xsmooth[:,2],'r',label='$b$')
    ax2.plot(time,Xsmooth[:,3],'g',label='$c$')
    ax2.plot(time,Xsmooth[:,4],'k',label='$d$')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('parameters')
    ax2.set_xlim([0, time[-1]])
    ax2.legend()
    ax2.grid(True)
    
    plt.show()