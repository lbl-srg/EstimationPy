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
    filePath = "../../modelica/FmuExamples/Resources/FMUs/HeatExchanger.fmu"
    
    # ReInit the model with the new FMU
    m.ReInit(filePath, atol=1e-5, rtol=1e-6)
    
    # Show details
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Set the CSV file associated to the input
    inputPath = "../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    input = m.GetInputByName("mFlow_cold")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.mFlow_COLD")
        
    inputPath = "../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    input = m.GetInputByName("mFlow_hot")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.mFlow_HOT")
        
    inputPath = "../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    input = m.GetInputByName("T_hot")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.Thot_IN")
        
    inputPath = "../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    input = m.GetInputByName("T_cold")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.Tcold_IN")
    
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
        if "__" not in name:
            pylab.subplot(N,1,i)
            pylab.plot(time, values, label=name)
            pylab.ylabel(name)
            pylab.xlabel('Time')
            i += 1
    pylab.show()
   
if __name__ == '__main__':
    main()