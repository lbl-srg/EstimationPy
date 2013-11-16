'''
Created on Nov 7, 2013

@author: marco
'''
import pylab
from FmuUtils import Model

def main():
    
    # Initialize the FMU model empty
    m = Model.Model()
    
    # Assign an existing FMU to the model
    filePath = "../../modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu"
    
    # ReInit the model with the new FMU
    m.ReInit(filePath)
    
    # Show details
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Set the CSV file associated to the input
    inputPath = "../../modelica/FmuExamples/Resources/data/SimulationData_FirstOrder.csv"
    input = m.GetInputByName("u")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("system.u")
    
    
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