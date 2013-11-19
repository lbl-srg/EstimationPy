'''
Created on Nov 7, 2013

@author: marco
'''
import pylab
import numpy
from FmuUtils import Model
from FmuUtils import CsvReader
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
    par.SetInitialValue(1000.0)
    par.SetCovariance(50.0)
    par.SetMinValue(50.0)
    par.SetConstraintLow(True)
    
    # show the info about the parameter to be identified
    print par.Info()
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("G_cold"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[1]
    par.SetInitialValue(1000.0)
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
    time, x, sqrtP, y, Sy = ukf_FMU.filter(0.0, 5.0)
    
    # Path of the csv file containing the True data series
    csvTrue = "../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, csvTrue)

def showResults(time, x, sqrtP, y, Sy, csvTrue):
    # Display results
    fig1 = pylab.figure()
    pylab.clf()
    
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("heatExchanger.metal.T")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d = numpy.squeeze(numpy.asarray(res["data"]))
    
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    
    pylab.subplot(2,1,1)
    pylab.plot(time, x[:,0],"r--")
    pylab.plot(t, d, "g")
    pylab.ylabel("x")
    pylab.xlabel('Time')
    
    #pylab.subplot(2,1,2)
    #pylab.plot(time, y)
    #pylab.ylabel("x")
    #pylab.xlabel('Time')
    
    pylab.show()
  
if __name__ == '__main__':
    main()