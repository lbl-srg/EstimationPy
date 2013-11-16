'''
Created on Nov 7, 2013

@author: marco
'''
import pylab
from FmuUtils import Model
from ukf.ukfFMU import ukfFMU

def main():
    
    # Assign an existing FMU to the model
    filePath = "../../modelica/FmuExamples/Resources/FMUs/HeatExchanger.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-5, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = "../../modelica/FmuExamples/Resources/data/NoisySimulationData_HeatExchanger.csv"
    
    ###################################################################
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("mFlow_cold")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.mFlow_COLD")
    input.SetCovariance(2.0)
    
    # Set the CSV file associated to the input, and its covariance 
    input = m.GetInputByName("mFlow_hot")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.mFlow_HOT")
    input.SetCovariance(2.0)
    
    # Set the CSV file associated to the input, and its covariance  
    input = m.GetInputByName("T_hot")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.Thot_IN")
    input.SetCovariance(1.0)
    
    # Set the CSV file associated to the input, and its covariance    
    input = m.GetInputByName("T_cold")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.Tcold_IN")
    input.SetCovariance(1.0)
    
    #################################################################
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("Tcold_OUT")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("heatExchanger.Tcold_OUT")
    output.SetMeasuredOutput()
    output.SetCovariance(1.0)
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("Thot_OUT")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("heatExchanger.Thot_OUT")
    output.SetMeasuredOutput()
    output.SetCovariance(1.0)
    
    #################################################################
    # Select the states to be estimated
    m.AddVariable(m.GetVariableObject("metal.T"))
    
    # Set initial value of state, and its covariance and the limits (if any)
    var = m.GetVariables()[0]
    var.SetInitialValue(328.15)
    var.SetCovariance(1.5)
    var.SetMinValue(273.15)
    var.SetConstraintLow(True)
    
    # show the info about the variable to be estimated
    print var.Info()
    
    #################################################################
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("G_hot"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[0]
    par.SetInitialValue(100.0)
    par.SetCovariance(50.0)
    par.SetMinValue(50.0)
    par.SetConstraintLow(True)
    
    # show the info about the parameter to be identified
    print par.Info()
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("G_cold"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[1]
    par.SetInitialValue(150.0)
    par.SetCovariance(50.0)
    par.SetMinValue(50.0)
    par.SetConstraintLow(True)
    
    # show the info about the parameter to be identified
    print par.Info()
    
    #################################################################
    # Initialize the model for the simulation
    #print "Before initialization: ", m.GetState()
    #print "State observed:",m.GetStateObservedValues()
    #print "Parameters estimated:",m.GetParametersValues()
    m.InitializeSimulator()
    print "After initialization: ", m.GetState()
    print "State observed:",m.GetStateObservedValues()
    print "Parameters estimated:",m.GetParametersValues()
    
    #################################################################
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