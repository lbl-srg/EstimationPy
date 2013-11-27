'''
Created on Nov 7, 2013

@author: marco
'''
import numpy
from FmuUtils import Model
from FmuUtils import CsvReader
from ukf.ukfFMU import ukfFMU

import matplotlib.pyplot as plt
from pylab import figure

def main():
    
    # Assign an existing FMU to the model
    filePath = "../../modelica/FmuExamples/Resources/FMUs/FmuExamples_ValveStuck_Quad.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-5, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = "../../modelica/FmuExamples/Resources/data/NoisyData_StuckValve_quad_noDyn.csv"
    
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
    output.SetCovariance(0.05)
    
    #################################################################
    # Select the parameter to be identified
    m.AddVariable(m.GetVariableObject("command.y"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    var = m.GetVariables()[0]
    var.SetInitialValue(0.8)
    var.SetCovariance(0.05)
    var.SetMinValue(0.0)
    var.SetConstraintLow(True)
    var.SetMaxValue(1.05)
    var.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    #ukf_FMU.setUKFparams()
    
    # start filter
    time, x, sqrtP, y, Sy = ukf_FMU.filter(0.0, 5.0, verbose=False)
    
    # Path of the csv file containing the True data series
    csvTrue = "../../modelica/FmuExamples/Resources/data/SimulationData_StuckValve_quad_noDyn.csv"
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, csvTrue, m)

def showResults(time, x, sqrtP, y, Sy, csvTrue, m):
    # Convert list to arrays
    x = numpy.squeeze(numpy.array(x))
    y = numpy.squeeze(numpy.array(y))
    sqrtP = numpy.squeeze(numpy.array(sqrtP))
    Sy = numpy.squeeze(numpy.array(Sy))
    
    ####################################################################
    # Display results
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("valveStuck.m_flow")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d = numpy.squeeze(numpy.asarray(res["data"]))
    
    outputRes = m.GetOutputByName("m_flow").GetCsvReader()
    outputRes.SetSelectedColumn("valveStuck.m_flow")
    res = outputRes.GetDataSeries()
    
    to = res["time"]
    do = numpy.squeeze(numpy.asarray(res["data"]))
    
    fig0 = plt.figure()
    fig0.set_size_inches(12,8)
    ax0  = fig0.add_subplot(111)
    ax0.plot(t,d,'g',label='$\dot{m}_{FLOW}^{Model}$',alpha=1.0)
    ax0.plot(to,do,'go',label='$\dot{m}_{FLOW}^{Noisy}$',alpha=0.5)
    ax0.plot(time,y,'r',label='$\dot{m}_{FLOW}^{Filter}$')
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('Mass flow rate [kg/s]')
    ax0.set_xlim([t[0], t[-1]])
    ax0.set_ylim([0, 3])
    legend = ax0.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax0.grid(False)
    plt.savefig('Flow.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results
    
    simResults.SetSelectedColumn("valveStuck.valve.opening")
    res = simResults.GetDataSeries()
    opening = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("valveStuck.cmd")
    res = simResults.GetDataSeries()
    command = numpy.squeeze(numpy.asarray(res["data"]))
    
    fig1 = plt.figure()
    fig1.set_size_inches(12,8)
    ax1  = fig1.add_subplot(111)
    ax1.plot(t,command,'g',label='$Cmd$',alpha=1.0)
    ax1.plot(t,opening,'b',label='$Pos$',alpha=1.0)
    ax1.plot(time,x,'r',label='$Pos^{UKF}$')
    ax1.fill_between(time, x - sqrtP, x + sqrtP, facecolor='red', interpolate=True, alpha=0.3)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Valve opening [$\cdot$]')
    ax1.set_xlim([t[0], t[-1]])
    ax1.set_ylim([0, 1.1])
    legend = ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    plt.savefig('Positions.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    plt.show()
    
if __name__ == '__main__':
    main()