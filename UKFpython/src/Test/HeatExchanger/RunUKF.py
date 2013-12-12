'''
Created on Nov 7, 2013

@author: marco
'''

import numpy
import matplotlib.pyplot as plt

from FmuUtils import Model
from FmuUtils import CsvReader
from ukf.ukfFMU import ukfFMU

def main():
    
    # Assign an existing FMU to the model
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/HeatExchanger.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-5, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = "../../../modelica/FmuExamples/Resources/data/NoisySimulationData_HeatExchanger.csv"
    
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
    var.SetInitialValue(310.15)
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
    time, x, sqrtP, y, Sy, y_full  = ukf_FMU.filter(0.0, 5.0)
    
    # Path of the csv file containing the True data series
    csvTrue = "../../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, y_full, csvTrue)

def showResults(time, x, sqrtP, y, Sy, y_full, csvTrue):
    
    # convert results
    x = numpy.array(x)
    y = numpy.array(y)
    sqrtP = numpy.array(sqrtP)
    Sy = numpy.array(Sy)
    y_full = numpy.squeeze(numpy.array(y_full))
    
    # Get the true and unmeasured metal temperature
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("heatExchanger.metal.T")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d = numpy.squeeze(numpy.asarray(res["data"]))
    
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(211)
    id = 0
    ax1.plot(time,x[:,id],'r',label='$\hat{T}_{Metal}$',alpha=1.0)
    ax1.fill_between(time, x[:,id] - sqrtP[:,id,id], x[:,id] + sqrtP[:,id,id], facecolor='red', interpolate=True, alpha=0.3)
    ax1.plot(t,d,'g',label='$T_{Metal}$',alpha=1.0)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Metal temperature')
    ax1.set_xlim([time[0], time[-1]])
    legend = ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    
    ax2  = fig1.add_subplot(212)
    id = 1
    ax2.plot(time,x[:,id],'b',label='$\hat{G}_{Hot}$',alpha=1.0)
    ax2.fill_between(time, x[:,id] - sqrtP[:,id,id], x[:,id] + sqrtP[:,id,id], facecolor='blue', interpolate=True, alpha=0.3)
    id = 2
    ax2.plot(time,x[:,id],'c',label='$\hat{G}_{Cold}$',alpha=1.0)
    ax2.fill_between(time, x[:,id] - sqrtP[:,id,id], x[:,id] + sqrtP[:,id,id], facecolor='cyan', interpolate=True, alpha=0.3)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Thermal conductances')
    ax2.set_xlim([t[0], t[-1]])
    legend = ax2.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    plt.savefig('Heat_Exchanger_UKF.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    plt.show()
    
    plt.show()
  
if __name__ == '__main__':
    main()