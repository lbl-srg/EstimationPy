'''
Created on Sep 18, 2013

@author: marco
'''
import numpy as np
import pyfmi

from ukf.Model import Model
from ukf.ukf import ukf

class model(Model):
    
    """
    Initialize the model with parameters
    """
    def __init__(self, X0, pars = None):
        # use the initialization method of inherited class
        # there is  1 state, and it is observed
        # there are 2 parameters to be identified
        # there are 2 output variables
        Model.__init__(self, 1, 1, 4, 2, X0, pars, [1.0], [1.0, 1.0])
        
        # Load the FMU
        self.FMUmodel = pyfmi.fmi.load_fmu('/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu')
        
        # Force the initialization of the state variables in the fmu model
        self.setInitialState(X0)
        
        # set simulate options
        self.opts = self.FMUmodel.simulate_options()
        self.opts["initialize"] = True
        
    """
    This method modify the parameters of the model
    """
    def setPars(self, pars):
        # parameters of the model
        try:
            (r, c) = np.shape(pars)
            temp = np.array(pars[0:self.n_pars])
            p = temp[0]
        except ValueError:
            (r, ) = np.shape(pars)
            p = pars
        
        self.FMUmodel.set('a', p[0])
        self.FMUmodel.set('b', p[1])
        self.FMUmodel.set('c', p[2])
        self.FMUmodel.set('d', p[3])
    
    """
    This method returns the parameters of the model
    """
    def getPars(self):
        # parameters of the model
        a = self.FMUmodel.get('a')
        b = self.FMUmodel.get('b')
        c = self.FMUmodel.get('c')
        d = self.FMUmodel.get('d')
        return np.hstack((a, b, c, d))
      
    """
    This method set the value of the state variable x
    """
    def setInitialState(self, X0):
        Model.setInitialState(self, X0)
        try:
            (r, c) = np.shape(X0)
            x = np.array(X0[0:self.n_states_obs])
        except ValueError:
            (r, ) = np.shape(X0)
            c = None
            x = np.array([X0])
        self.FMUmodel._set_continuous_states(x[0])
    
    """
    This method get the value of the state variable x
    """
    def getState(self):
        return np.array(self.FMUmodel.continuous_states)
    
    """
    This method get the value of the augmented state variable x
    """
    def getAugState(self):
        x = np.array(self.FMUmodel.continuous_states)
        par = self.getPars()
        return np.hstack((x,par))
    
    """
    This method set the value of the states and parameters that are part of the augmented state
    """
    def setAugState(self, Xaug):
        try:
            (r, c) = np.shape(Xaug)
            self.setInitialState(Xaug[0,0:self.n_states_obs])
            self.setPars(Xaug[0,self.n_states_obs:self.n_states_obs+self.n_pars])
        except ValueError:
            self.setInitialState(Xaug[0:self.n_states_obs])
            self.setPars(Xaug[self.n_states_obs:self.n_states_obs+self.n_pars])
    
    """
    State evolution function

    * x: state vector at time t
    * u: input vector at time t
    * t: time

    it returns

    * new state at time t+1
    """
    def functionF(self, val):
        (x, u_old, u, t_old, t, simulate) = val
        t     = np.float(t)
        u     = np.float(u)
        t_old = np.float(t_old)
        u_old = np.float(u_old)
        
        # Define trajectory
        time = np.array([t_old, t])
        input = np.array([u_old, u])
        u_traj  = np.transpose(np.vstack((time, input)))
        
        # Create input object
        input_object = (['u'],u_traj)
        
        # set the state values
        self.setInitialState(x[0:self.n_states])
        
        # set the parameters
        self.setPars(x[self.n_states:self.n_states+self.n_pars])
        
        # simulate
        res = self.FMUmodel.simulate(start_time = t_old, input = input_object, final_time = t, options = self.opts)
        self.opts["initialize"] = False
        
        # get the output and add the noise
        y1 = res['y'][-1]
        y2 = res['x'][-1]
        y  = np.array([y1, y2])
        
        # get the last value of the state and add the noise 
        x_new = self.getAugState()
        
        # get the process and output noise
        if len(x)>self.n_states+self.n_pars:
            q = x[self.n_states+self.n_pars:self.n_states+self.n_pars+self.n_states_obs]
            r = x[self.n_states+self.n_pars+self.n_states_obs:]
            y += r
            x_new[0:self.n_states_obs] += q
        
        # return the state
        return x_new, y
    
    
if __name__ == '__main__':
    """
    Run the program
    """
    from utilities.getCsvData import getCsvData
    # import data from CSV file
    dataMatrix = getCsvData("/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/data/NoisySimulationData_FirstOrder.csv")

    # time, input and output vectors (from CSV file)
    time = dataMatrix[:,0]
    U = dataMatrix[:,1]
    Z = dataMatrix[:,3]
    
    x0 = 2.5
    a = -1
    b = 2.5
    c = 3.0
    d = 0.1
    Q = np.diag([1.0])
    R = np.diag([1.0])
    X0 = np.array([x0])
    pars = np.array([ a, b, c, d])
    
    m = model(X0)
    m.setPars(pars)
    m.setQ(Q)
    m.setR(R)
    sqrtQ = m.getSqrtQ()
    sqrtR = m.getSqrtR()
    
    # use the filter
    filter = ukf(1, 1, 4, 1)
    #filter.setUKFparams(0.1, 2, 1)
    
    wm, wc = filter.getWeights()
    
    sqrtP = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
    
    Xs = filter.computeSigmaPoints(m.getState(), m.getPars(), sqrtP)
    
    Xs_new, Ys = filter.sigmaPointProj(m, Xs, U[0], U[1], time[0], time[1])
    
    Xs_new_avg = filter.averageProj(Xs_new)
    Ys_avg = filter.averageOutProj(Ys)
    
    Cxx = filter.computeP(Xs_new, Xs_new_avg, Q)
    print "Cxx"
    print Cxx
    
    Sxx = filter.computeS(Xs_new, Xs_new_avg, sqrtQ)
    print "Sxx"
    print Sxx
    print "Error Cxx - Sxx**2"
    print np.dot(Sxx, Sxx) - Cxx
    
    Cyy = filter.computeCovZ(Ys, Ys_avg, R)
    print "Cyy"
    print Cyy
    
    Syy = filter.computeSy(Ys, Ys_avg, sqrtR)
    print "Syy"
    print Syy
    print "Error Cyy - Syy**2"
    print np.dot(Syy, Syy) - Cyy
    
    Cxy = filter.computeCovXZ(Xs_new, Xs_new_avg, Ys, Ys_avg)
    print "Cxy"
    print Cxy
    
    L = np.dot(Cxy, np.linalg.inv(Cyy))
    print "L"
    print L
    
    xnew = Xs_new_avg + np.dot(Z[1] - Ys_avg, np.transpose(L) )
    print "Xnew corrected"
    print xnew
    
    Cxx_new = Cxx - np.dot(np.dot(L, Cyy), np.transpose(L))
    print "Cxx corrected"
    print Cxx_new
    
    m.setAugState(xnew[0,:])
    