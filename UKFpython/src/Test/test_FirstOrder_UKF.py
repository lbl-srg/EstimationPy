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
    
    # Select the states to be identified, and add it to the list
    m.AddVariable(m.GetVariableObject("x"))
    
    # Set initial value of state, and its covariance and the limits (if any)
    var = m.GetVariables()[0]
    var.SetInitialValue(7.0)
    var.SetCovariance(0.5)
    var.SetMinValue(0.0)
    var.SetConstraintLow(True)
    
    #################################################################
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("b"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[0]
    par.SetInitialValue(5.0)
    par.SetCovariance(1.0)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    
    # show the info about the variable to be estimated
    print var.Info()
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    
    # start filter
    time, x, sqrtP, y, Sy = ukf_FMU.filter(0.0, 5.0, verbose=False)
    
    # Path of the csv file containing the True data series
    csvTrue = "../../modelica/FmuExamples/Resources/data/SimulationData_FirstOrder.csv"
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, csvTrue)

def showResults(time, x, sqrtP, y, Sy, csvTrue):
    # Display results
    fig1 = pylab.figure()
    pylab.clf()
    
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("system.x")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d = numpy.squeeze(numpy.asarray(res["data"]))
    
    pylab.subplot(2,1,1)
    pylab.plot(time, x,"r--")
    pylab.plot(t, d, "g")
    pylab.ylabel("x")
    pylab.xlabel('Time')
    
    pylab.subplot(2,1,2)
    pylab.plot(time, y)
    pylab.ylabel("x")
    pylab.xlabel('Time')
    
    pylab.show()
  
if __name__ == '__main__':
    main()