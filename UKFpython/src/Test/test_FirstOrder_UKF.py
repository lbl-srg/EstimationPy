'''
Created on Nov 7, 2013

@author: marco
'''
import pylab
from FmuUtils import Model

def main():
    
    # Assign an existing FMU to the model
    filePath = "../../modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath)
    
    # Path of the csv file containing the data series
    csvPath = "../../modelica/FmuExamples/Resources/data/NoisySimulationData_FirstOrder.csv" 
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("u")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("system.u")
    input.SetCovariance(2.0)
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("y")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("system.y")
    output.SetCovariance(2.0)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # Select the states to be identified, and add it to the list
    m.AddVariable(m.GetVariableObject("x"))
    
    # Set initial value of state, and its covariance
    var = m.GetVariables()[0]
    var.SetInitialValue(2.0)
    var.SetCovariance(0.5)
    
    # show the info about the variable to be estimated
    print var.Info()
    
    return

    # Instantiate filter, and run it
                        
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
        pylab.subplot(N,1,i)
        pylab.plot(time, values, label=name)
        pylab.ylabel(name)
        pylab.xlabel('Time')
        i += 1
    pylab.show()
   
if __name__ == '__main__':
    main()