'''
Created on Nov 7, 2013

@author: marco
'''
import pylab
import numpy
from FmuUtils import Model
from FmuUtils.FmuPool import FmuPool

def main():
    
    # Assign an existing FMU to the model
    filePath = "../../modelica/FmuExamples/Resources/FMUs/Pump_MBL2.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-4, rtol=1e-4)
    
    # SHow details of the model
    print m
    
    # Path of the csv file containing the data series
    csvPath = "../../modelica/FmuExamples/Resources/data/DataPumpShort.csv"
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("Nrpm")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("Pump.Speed")
    
    # Select the states to be modified
    m.AddParameter(m.GetVariableObject("pump.power.P[1]"))
    m.AddParameter(m.GetVariableObject("pump.power.P[2]"))
    m.AddParameter(m.GetVariableObject("pump.power.P[3]"))

    # Initialize the simulator
    m.InitializeSimulator()
    
    # Instantiate the pool
    pool = FmuPool(m, debug = True)

    # define the vector of initial conditions for which the simulations
    # have to be performed.
    # values has to be a list of state vectors
    # values = [ [x0_0], [x0_1], ... [x0_n]]
    vectorValues = numpy.linspace(0.2, 1.0, 5)
    values = []
    for v in vectorValues:
        temp = {"state":[], "parameters":numpy.array([v,v,v])}
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
        N = 2
        
        for name, values in results.iteritems():
            if "__" not in name:
                pylab.subplot(N,1,i)
                pylab.plot(time, values, "grey")
                pylab.ylabel(name)
                pylab.xlabel('Time')
                i += 1
            
    pylab.show()
    
if __name__ == '__main__':
    main()