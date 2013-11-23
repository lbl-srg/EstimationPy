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
    filePath = "../../modelica/FmuExamples/Resources/FMUs/Pump.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-3, rtol=1e-3)
    
    # Path of the csv file containing the data series
    csvPath = "../../modelica/FmuExamples/Resources/data/DataPump2.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("Nrpm")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("Pump.Speed")
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("P_el")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("Pump.kW")
    output.SetMeasuredOutput()
    output.SetCovariance(0.5)
    
    #################################################################
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("b"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[0]
    par.SetInitialValue(0.0)
    par.SetCovariance(0.1)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("b1"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[1]
    par.SetInitialValue(1.0)
    par.SetCovariance(0.1)
    par.SetMinValue(1.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(3.0)
    par.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    
    # start filter
    time, x, sqrtP, y, Sy = ukf_FMU.filter(0.0, 5.0, verbose=False)
     
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, csvPath)

def showResults(time, x, sqrtP, y, Sy, csvTrue):
    # Display results
    fig1 = pylab.figure()
    pylab.clf()
    
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("Pump.kW")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d_kW = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("Pump.Speed")
    res = simResults.GetDataSeries()
    d_rpm = numpy.squeeze(numpy.asarray(res["data"]))
    
    pylab.subplot(3,1,1)
    pylab.plot(time, x,"r--")
    pylab.ylabel("x")
    pylab.xlabel('Time')
    
    pylab.subplot(3,1,2)
    pylab.plot(t,d_kW,"g")
    pylab.plot(time, y,"r--")
    pylab.ylabel("y")
    pylab.xlabel('Time')
    
    
    a = x[-1][0]
    a1 = x[-1][1]
    
    a_av = numpy.average(x[:][0])
    a1_av = numpy.average(x[:][1])
    
    print a, a1
    print a_av, a1_av
    
    rpm = numpy.linspace(35.0,60.0,100)
    x_n = rpm/60.0
    app = 4.4*(a*x_n*x_n + (1 - a)*x_n)**a1
    app_av = 4.4*(a_av*x_n*x_n + (1 - a_av)*x_n)**a1_av
    
    
    pylab.subplot(3,1,3)
    pylab.plot(d_rpm,d_kW,"bo")
    pylab.plot(rpm, app,"r")
    #pylab.plot(rpm, app_av,"g")
    pylab.ylabel("kW")
    pylab.xlabel('rpm')
    
    
    
    pylab.show()
  
if __name__ == '__main__':
    main()