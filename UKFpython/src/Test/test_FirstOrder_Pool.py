'''
Created on Nov 7, 2013

@author: marco
'''

import pylab
import numpy
    
from FmuUtils.FmuPool import FmuPool
from FmuUtils.Model import Model

def main():

    # Initialize the FMU model empty
    m = Model()

    # Assign an existing FMU to the model
    filePath = "../../modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu"
    m.ReInit(filePath)

    # Set the CSV file associated to the input
    inputPath = "../../modelica/FmuExamples/Resources/data/SimulationData_FirstOrder.csv"
    input = m.GetInputByName("u")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("system.u")
    
    # Select the states to be modified
    m.AddVariable(m.GetVariableObject("x"))

    # Initialize the simulator
    m.InitializeSimulator()

    # Instantiate the pool
    pool = FmuPool(m, debug = False)

    # define the vector of initial conditions for which the simulations
    # have to be performed.
    # values has to be a list of state vectors
    # values = [ [x0_0], [x0_1], ... [x0_n]]
    vectorValues = numpy.linspace(1.0, 5.0, 1000)
    values = []
    for v in vectorValues:
        temp = {"state":numpy.array([v]), "parameters":[]}
        values.append(temp)
    
    # Run simulations in parallel
    poolResults = pool.Run(values)
    
    # plot all the results
    showResults(poolResults)    
    

def showResults(poolResults):
    
    # Display results
    fig1 = pylab.figure()
    pylab.clf()
    
    for res in poolResults:
        # get the results of a worker of the pool
        time, results = res[0]
    
        i = 1
        N = len(results.keys())
        for name, values in results.iteritems():
            pylab.subplot(N,1,i)
            pylab.plot(time, values, "grey")
            pylab.ylabel(name)
            pylab.xlabel('Time')
            i += 1
            
    pylab.show()
    
if __name__ == '__main__':
    main()