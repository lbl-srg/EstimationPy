'''
Created on Nov 7, 2013

@author: marco
'''
import pylab
from FmuUtils import Model
from ukf.ukfFMU import ukfFMU

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
    output.SetMeasuredOutput()
    output.SetCovariance(2.0)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # Select the states to be identified, and add it to the list
    m.AddVariable(m.GetVariableObject("x"))
    
    # Set initial value of state, and its covariance and the limits (if any)
    var = m.GetVariables()[0]
    var.SetInitialValue(2.0)
    var.SetCovariance(0.5)
    var.SetMinValue(0.0)
    var.SetConstraintLow(True)
    
    # show the info about the variable to be estimated
    print var.Info()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    
    # Show details
    print ukf_FMU
    
    # start filter
    ukf_FMU.filter(0.0, 5.0)
    
    return

    # Instantiate filter, and run it
   
if __name__ == '__main__':
    main()