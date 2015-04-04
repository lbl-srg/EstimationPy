'''
Created on Nov 7, 2013

@author: marco
'''
import pylab
import numpy
from FmuUtils import Model, Strings
from FmuUtils import CsvReader
from ukf.ukfFMU import ukfFMU

def main():
    
    # Initialize the FMU model empty
    m = Model.Model()
    
    # Assign an existing FMU to the model
    #filePath = "../../../modelica/FmuExamples/Resources/FMUs/Fmu_ValveStuck_bias3.fmu"
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/FmuValveSimple.fmu"
    
    # ReInit the model with the new FMU
    m.ReInit(filePath, verbose=Strings.SOLVER_VERBOSITY_LOUD)
    
    # SHow details of the model
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Path of the csv file containing the data series
    csvPath = "../../../modelica/FmuExamples/Resources/data/NoisyData_CalibrationValve_noDrift.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("dp")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("valveStuck.dp")
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("cmd")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("valveStuck.cmd")
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("T_in")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("valveStuck.T_in")
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # Simulate
    time, results = m.Simulate()
    
    # Show the results
    showResults(time, results)

def showResults(time, results):
    # Display results
    fig1 = pylab.figure()
    pylab.clf()
    i = 1
    N = len(results.keys())
    for name, values in results.iteritems():
        if "__"not in name:
            pylab.subplot(N,1,i)
            pylab.plot(time, values, label=name)
            pylab.ylabel(name)
            pylab.xlabel('Time')
            i += 1
    pylab.show()
  
if __name__ == '__main__':
    main()