'''
Created on Nov 7, 2013

@author: marco
'''
import os
import platform
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from FmuUtils import Model
from FmuUtils import CsvReader
from ukf.ukfFMU import ukfFMU

def main():
    
    # Assign an existing FMU to the model, depending on the platform identified
    dir_path = os.path.dirname(__file__)
    
    # Define the path of the FMU file
    if platform.architecture()[0]=="32bit":
        print "32-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder.fmu")
    else:
        print "64-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_64bit.fmu")
        
    # Initialize the FMU model empty
    m = Model.Model(filePath)
    
    # Path of the csv file containing the data series
    csvPath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "data", "NoisySimulationData_FirstOrder.csv")
    
    # Set the CSV file associated to the input, and its covariance
    input_u = m.GetInputByName("u")
    input_u.GetCsvReader().OpenCSV(csvPath)
    input_u.GetCsvReader().SetSelectedColumn("system.u")
    input_u.SetCovariance(2.0)
    
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
    var.SetInitialValue(1.5)
    var.SetCovariance(0.5)
    var.SetMinValue(0.0)
    var.SetConstraintLow(True)
    
    # show the info about the variable to be estimated
    print var.Info()
    
    # Set parameters been identified
    par_a = m.GetVariableObject("a")
    m.SetReal(par_a, -0.90717055)
    par_b = m.GetVariableObject("b")
    m.SetReal(par_b, 2.28096907)
    par_c = m.GetVariableObject("c")
    m.SetReal(par_c, 3.01419707)
    par_d = m.GetVariableObject("d")
    m.SetReal(par_d, 0.06112703)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    
    # Start the filter
    t0 = pd.to_datetime(0.0, unit = "s")
    t1 = pd.to_datetime(10.0, unit = "s")
    time, x, sqrtP, y, Sy, y_full = ukf_FMU.filter(start = t0, stop = t1, verbose=False)
    
    # Path of the csv file containing the True data series
    csvTrue = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, y_full, csvTrue, csvPath, m)

def showResults(time, x, sqrtP, y, Sy, y_full, csvTrue, csvNoisy, m):
    # convert results
    x = numpy.array(x)
    y = numpy.array(y)
    sqrtP = numpy.array(sqrtP)
    Sy = numpy.array(Sy)
    y_full = numpy.squeeze(numpy.array(y_full))
    
    # Read from file
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvNoisy)
    
    simResults.SetSelectedColumn("system.x")
    res = simResults.GetDataSeries()
    t_n = res.index
    d_x_n = res.values
    
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    
    simResults.SetSelectedColumn("system.x")
    res = simResults.GetDataSeries()
    t = res.index
    d_x = res.values
    
    simResults.SetSelectedColumn("system.y")
    res = simResults.GetDataSeries()
    d_y = res.values
    
    simResults.SetSelectedColumn("system.u")
    res = simResults.GetDataSeries()
    d_u = res.values
    
    output = m.GetOutputByName("y")
    output.GetCsvReader().SetSelectedColumn("system.y")
    res = output.GetCsvReader().GetDataSeries()
    d_y_n = res.values
    
    input_u = m.GetInputByName("u")
    input_u.GetCsvReader().SetSelectedColumn("system.u")
    res = input_u.GetCsvReader().GetDataSeries()
    d_u_n = res.values
    
    fig0 = plt.figure()
    fig0.set_size_inches(12,8)
    ax0  = fig0.add_subplot(111)
    ax0.plot(t,d_x,'g',label='$x(t)$',alpha=1.0)
    ax0.plot(t_n,d_x_n,'go',label='$x_m(t)$',alpha=1.0)
    ax0.plot(time, x,'r',label='$\hat{x}(t)$')
    ax0.fill_between(time, x[:,0] - sqrtP[:,0,0], x[:,0] + sqrtP[:,0,0], facecolor='red', interpolate=True, alpha=0.3)
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('State Variable')
    ax0.set_xlim([t[0], t[-1]])
    legend = ax0.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax0.grid(False)
    plt.savefig('FirstOrder_State.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(212)
    ax1.plot(t_n,d_y_n,'go',label='$y_m(t)$',alpha=1.0)
    ax1.plot(t,d_y,'g',label='$y(t)$',alpha=1.0)
    ax1.plot(time, y,'r',label='$\hat{y}(t)$')
    ax1.fill_between(time, y[:,0] - Sy[:,0,0], y[:,0] + Sy[:,0,0], facecolor='red', interpolate=True, alpha=0.3)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Output Variable')
    ax1.set_xlim([t[0], t[-1]])
    legend = ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    
    ax2  = fig1.add_subplot(211)
    ax2.plot(t_n,d_u_n,'bo',label='$u_m(t)$',alpha=1.0)
    ax2.plot(t,d_u,'b',label='$u(t)$',alpha=1.0)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Input Variable')
    ax2.set_xlim([t[0], t[-1]])
    legend = ax2.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    plt.savefig('FirstOrder_InputOutput.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    plt.show()
    
if __name__ == '__main__':
    main()