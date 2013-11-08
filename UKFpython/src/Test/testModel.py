'''
Created on Nov 7, 2013

@author: marco
'''
import pylab
from FmuUtils import Model

def main():
    # This flag is used to choose the example
    FirstOrder = False
    
    # Initialize the FMU model empty
    m = Model.Model()
    
    # Assign an existing FMU to the model
    if FirstOrder:
        filePath = "../../modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu"
    else:
        filePath = "../../modelica/FmuExamples/Resources/FMUs/HeatExchanger.fmu"
    
    # ReInit the model with the new FMU
    m.ReInit(filePath)
    
    # Show details
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Set the CSV file associated to the input
    if FirstOrder:
        inputPath = "../../modelica/FmuExamples/Resources/data/SimulationData_FirstOrder.csv"
        input = m.GetInputByName("u")
        input.GetCsvReader().OpenCSV(inputPath)
        input.GetCsvReader().SetSelectedColumn("system.u")
    else:
        inputPath = "../../modelica/FmuExamples/Resources/data/NoisySimulationData_HeatExchanger.csv"
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
    showResults(FirstOrder, time, results)
    
    

def showResults(FirstOrder, time, results):
    if True:
        # Display results
        pylab.figure()
        pylab.clf()
        i = 1
        for name, values in results.iteritems():
            pylab.subplot(6,1,i)
            pylab.plot(time, values, "")
            pylab.xlabel('time')
            i += 1
            
        pylab.show()
   
if __name__ == '__main__':
    main()