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
    filePath = "../../modelica/FmuExamples/Resources/FMUs/FmuExamples_ValveStuck.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-5, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = "../../modelica/FmuExamples/Resources/data/NoisySimulationData_StuckValve.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("dp")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("valveStuck.dp")
    input.SetCovariance(1000.0)
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("cmd")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("valveStuck.cmd")
    input.SetCovariance(0.0)
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("m_flow")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("valveStuck.m_flow")
    output.SetMeasuredOutput()
    output.SetCovariance(0.01)
    
    #################################################################
    # Select the parameter to be identified
    m.AddVariable(m.GetVariableObject("command.y"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    var = m.GetVariables()[0]
    var.SetInitialValue(0.8)
    var.SetCovariance(0.5)
    var.SetMinValue(0.0)
    var.SetConstraintLow(True)
    var.SetMaxValue(1.5)
    var.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    ukf_FMU.setUKFparams()
    
    # start filter
    time, x, sqrtP, y, Sy = ukf_FMU.filter(0.0, 5.0, verbose=False)
    
    # Path of the csv file containing the True data series
    csvTrue = "../../modelica/FmuExamples/Resources/data/SimulationData_StuckValve.csv"
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, csvTrue)

def showResults(time, x, sqrtP, y, Sy, csvTrue):
    # Display results
    fig1 = pylab.figure()
    pylab.clf()
    
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("valveStuck.m_flow")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d = numpy.squeeze(numpy.asarray(res["data"]))
    
    pylab.subplot(2,1,1)
    pylab.plot(time, y,"r--")
    pylab.plot(t, d, "g")
    pylab.ylabel("m_flow")
    pylab.xlabel('Time')
    
    simResults.SetSelectedColumn("valveStuck.limiter.y")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d = numpy.squeeze(numpy.asarray(res["data"]))
    
    pylab.subplot(2,1,2)
    pylab.plot(time, x)
    pylab.plot(t, d, "g")
    pylab.ylabel("x")
    pylab.xlabel('Time')
    
    pylab.show()
  
if __name__ == '__main__':
    main()