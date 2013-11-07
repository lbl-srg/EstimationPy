'''
Created on Nov 7, 2013

@author: marco
'''
import numpy
from FmuUtils import Model

def main():
    FirstOrder = False
    
    # Initialize the FMU model empty
    m = Model.Model()
    
    # Assign an existing FMU to the model
    if FirstOrder:
        filePath = "../../modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu"
    else:
        filePath = "../../modelica/FmuExamples/Resources/FMUs/HeatExchanger.fmu"
    m.ReInit(filePath)
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the inputs
    # print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
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
    
    # Show the details of the CsvReader
    # print input.GetCsvReader()
    
    # Get the data series
    #dataSeries = input.GetCsvReader().GetDataSeries()
    #if dataSeries == {}:
    #    return
    #time = dataSeries["time"]
    #input = dataSeries["data"]    
    
    # Initialize the model for the simulation
    #start_time = time[0]
    #start_input = numpy.matrix(input[0])
    #m.InitializeSimulator(start_time, start_input)
    m.InitializeSimulator()
                          
    # Simulate
    #m.Simulate(time, input)
    
    
    
if __name__ == '__main__':
    main()