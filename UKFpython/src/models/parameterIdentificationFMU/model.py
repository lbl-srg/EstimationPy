'''
Created on Sep 18, 2013

@author: marco
'''
import numpy as np
from scipy import integrate
import pyfmi

from ukf.Model import Model

class model(Model):
    
    """
    Initialize the model with parameters
    """
    def __init__(self, X0 = np.array([4.0, -1.0, 2.5, 3.0, 0.1])):
        # use the initialization method of inherited class
        Model.__init__(self, 5, 5, 1, X0, None, [1.0], [1.0])
        
        # Load the FMU
        self.FMUmodel   = pyfmi.fmi.load_fmu('/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu')
        
        # Force the initialization of the state variables in the fmu model
        self.setInitialState(X0)
        
        # set parameters of the model
        self.setPars(X0[1], X0[2], X0[3], X0[4])
        
        # set simulate options
        self.opts = self.FMUmodel.simulate_options()
        self.opts["initialize"] = True
        
    """
    This method modify the parameters of the model
    """
    def setPars(self, a, b, c, d):
        # parameters of the model
        self.FMUmodel.set('a',a)
        self.FMUmodel.set('b',b)
        self.FMUmodel.set('c',c)
        self.FMUmodel.set('d',d)
    
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
        x = np.array([X0[0]])
        self.FMUmodel._set_continuous_states(x)
        self.setPars(X0[1], X0[2], X0[3], X0[4])
    
    """
    This method get the value of the state variable x
    """
    def getState(self):
        x = np.array(self.FMUmodel.continuous_states)
        par = self.getPars()
        return np.hstack((x,par))
    
    """
    State evolution function

    * x: state vector at time t
    * u: input vector at time t
    * t: time

    it returns

    * new state at time t+1
    """
    def functionF(self, val):
        (x,u_old,u,t_old,t,simulate) = val
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
        self.setInitialState(x)
        
        # simulate
        self.FMUmodel.simulate(start_time = t_old, input = input_object, final_time = t, options = self.opts)
        self.opts["initialize"] = False
        
        # get the last value
        x_new = self.getState()
        
        # return the state
        return x_new
    
    """
    Output measurement function

    * x: state vector at time t
    * u: input vector at time t
    * t: time

    it returns

    * output at time t
    """
    def functionG(self, x, u, t, simulate = True):
        # Define trajectory
        t = np.float(t)
        u = np.float(u)
        time = np.array([t, t + 1e-10])
        input = np.array([u, u])
        u_traj  = np.transpose(np.vstack((time, input)))
        
        # Create input object
        input_object = (['u'],u_traj)
        
        # set the state values
        self.setInitialState(x)
        
        # simulate
        res = self.FMUmodel.simulate(start_time = t, input = input_object, final_time = t+1e-10, options = self.opts)
        self.opts["initialize"] = False
        
        y = res['y'][-1]
        
        # return the output
        return y
    
    
if __name__ == '__main__':
    """
    Run the program
    """
    
    from utilities.getCsvData import getCsvData
    
    x0 = 2.5
    a = -1
    b = 2.5
    c = 3.0
    d = 0.1
    X0 = np.array([x0, a, b, c, d])
    m = model(X0)
    
    dataMatrix = getCsvData("/mnt/hgfs/Documents/Projects/ESTCP/eetd-estcp_ndw_eis/UKFpython/modelica/FmuExamples/Resources/data/NoisySimulationData_FirstOrder.csv")

    # time, input and output vectors (from CSV file)
    time = dataMatrix[:,0]
    U = dataMatrix[:,1]
    Z = dataMatrix[:,3]
    