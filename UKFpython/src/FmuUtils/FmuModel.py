'''
Created on Oct 17, 2013

@author: marco
'''

import pyfmi

class Model(object):
    
    def __init__(self, fmuName):
        self.SIMULATION_TRIES = 4

        self.model = pyfmi.load_fmu(fmuName)
        self.opts = self.model.simulate_options()
        # State variables of the model
        # the key indicates the order in the state vector, while the value is the valuereference in the FMU
        self.states = {}
        self.states[0] = 33554432
        # Parameters of the model
        # The key indicates the name of the parameter, while the values indicate the valuereference in the FMU
        self.parameters = {}
        self.parameters["x_start"] = 16777216
        self.parameters["a"] = 16777217
        self.parameters["f"] = 16777218
        

    def initSim(self):
        # This method perform a very short simulation to initialize the model
        # The next times the model will be simulated without the initialziation phase
        self.opts["initialize"] = True
        try:
            self.model.simulate(start_time = 0.0, final_time = 1.0e-10, options = self.opts)
            self.opts["initialize"] = False
            return True
        except ValueError:
            print "First simulation for initialize the model failed"
            return False
    
    def setResultFile(self, fileName):
        self.opts["result_file_name"] = fileName
    
    def setState(self, x0):
        try:
            N = len(x0)
            for i in range(N):
                valueref = self.states[i]
                self.model.set_real(valueref, x0[i])
        except TypeError:
            N = 1
            valueref = self.states[0]
            self.model.set_real(valueref, x0)
        
        return

    def getState(self):
        return self.model._get_continuous_states()

    def setParameters(self, pars):
        pass
    
    def getParameters(self):
        return self.model.get("x_start")

    def setAugmentedState(self, augState):
        pass

    def getAugmentedState(self):
        pass

    def simulate(self, start_time, final_time):
        simulated = False
        i = 0
        while not simulated and i < self.SIMULATION_TRIES:
            try:
                res = self.model.simulate(start_time = start_time, final_time = final_time, options = self.opts)
                simulated = True
            except ValueError:
                print "Simulation of the model failed, try again"
                i += 1
                res = None
                return res
        
        x_res = res['x']
        t     = res['time']
        
        return (t, x_res)