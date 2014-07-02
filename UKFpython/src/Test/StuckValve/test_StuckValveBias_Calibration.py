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

Kv2Av = 2.77e-5

def main():
    
    # Assign an existing FMU to the model
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/FmuValveSimple.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-5, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = "../../../modelica/FmuExamples/Resources/data/NoisyData_CalibrationValve_Drift.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("dp")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("valveStuck.dp")
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("cmd")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("valveStuck.cmd")
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("T_in")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("valveStuck.T_in")
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("m_flow")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("valveStuck.m_flow")
    output.SetMeasuredOutput()
    output.SetCovariance(0.03)
    
    #################################################################
    # Select the variable to be estimated
    m.AddParameter(m.GetVariableObject("valve.Av"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    var = m.GetParameters()[0]
    var.SetInitialValue(10.00*Kv2Av)
    var.SetCovariance(0.2*Kv2Av)
    var.SetMinValue(1.0*Kv2Av)
    var.SetConstraintLow(True)
    var.SetMaxValue(10.0*Kv2Av)
    var.SetConstraintHigh(True)
    
    m.AddParameter(m.GetVariableObject("lambda"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    var = m.GetParameters()[1]
    var.SetInitialValue(0.01)
    var.SetCovariance(0.0005)
    var.SetMinValue(-0.005)
    var.SetConstraintLow(True)
    var.SetMaxValue(0.025)
    var.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    ukf_FMU.setUKFparams(k=0)
    
    # start filter
    time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth = ukf_FMU.filterAndSmooth(0.0, 5.0, verbose=False)
    
    # Path of the csv file containing the True data series
    csvTrue = "../../../modelica/FmuExamples/Resources/data/SimulationData_CalibrationValve_Drift.csv"
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth, csvTrue, m)

def showResults(time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth, csvTrue, m):
    # Convert list to arrays
    x = numpy.array(x)
    y = numpy.array(y)
    sqrtP = numpy.array(sqrtP)
    Sy = numpy.array(Sy)
    y_full = numpy.squeeze(numpy.array(y_full))
    
    xs = numpy.array(Xsmooth)
    Ss = numpy.array(Ssmooth)
    Ys = numpy.array(Yfull_smooth)
    
    ####################################################################
    # Display results
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("valveStuck.m_flow_real")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d_real = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("valveStuck.Kv")
    res = simResults.GetDataSeries()
    
    d_Kv = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("valveStuck.lambda")
    res = simResults.GetDataSeries()
    
    d_lambda = numpy.squeeze(numpy.asarray(res["data"]))
    
    outputRes = m.GetOutputByName("m_flow").GetCsvReader()
    outputRes.SetSelectedColumn("valveStuck.m_flow")
    res = outputRes.GetDataSeries()
    
    to = res["time"]
    do = numpy.squeeze(numpy.asarray(res["data"]))
    
    fig0 = plt.figure()
    fig0.set_size_inches(12,8)
    ax0  = fig0.add_subplot(111)
    ax0.plot(t,d_real,'g',label='$\dot{m}$',alpha=1.0)
    ax0.plot(to,do,'go',label='$\dot{m}^{N+D}$',alpha=0.5)
    ax0.plot(time,y,'r',label='$\dot{m}^{Filter}$')
    ax0.plot(time,y_full[:,0],'r',label='$\hat{\dot{m}}$')
    ax0.plot(time,Ys[:,0],'b',label='$\hat{\dot{m}}^S$')
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('Mass flow rate [kg/s]')
    ax0.set_xlim([t[0], t[-1]])
    ax0.set_ylim([0, 1.3])
    legend = ax0.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax0.grid(False)
    plt.savefig('Flow.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results
    
    outputRes = m.GetInputByName("cmd").GetCsvReader()
    outputRes.SetSelectedColumn("valveStuck.cmd")
    res = outputRes.GetDataSeries()
    
    to = res["time"]
    do = numpy.squeeze(numpy.asarray(res["data"]))
    
    fig3 = plt.figure()
    idx = 0
    fig3.set_size_inches(12,8)
    ax3  = fig3.add_subplot(111)
    ax3.plot(t,d_Kv*Kv2Av,'g',label='$K_v$')
    ax3.plot(time,x[:,idx],'r',label='$\hat{K_v}$')
    ax3.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='red', interpolate=True, alpha=0.3)
    ax3.plot(time,xs[:,idx],'b',label='$\hat{K_v}^S$')
    ax3.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='blue', interpolate=True, alpha=0.3)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Valve $K_v$ coeff. [$m^3/hour$]')
    ax3.set_xlim([t[0], t[-1]])
    ax3.set_ylim([0, 10.0*Kv2Av])
    legend = ax3.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax3.grid(False)
    plt.savefig('ValveCoeff.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    fig4 = plt.figure()
    idx = 1
    fig4.set_size_inches(12,8)
    ax4  = fig4.add_subplot(111)
    ax4.plot(t,d_lambda,'g',label='$\lambda$')
    ax4.plot(time,x[:,idx],'r',label='$\hat{\lambda}$')
    ax4.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='red', interpolate=True, alpha=0.3)
    ax4.plot(time,xs[:,idx],'b',label='$\hat{\lambda}^S$')
    ax4.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='blue', interpolate=True, alpha=0.3)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Sensor thermal drift coeff. [$1/K$]')
    ax4.set_xlim([t[0], t[-1]])
    ax4.set_ylim([-0.0025, 0.015])
    legend = ax4.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax4.grid(False)
    plt.savefig('Drift.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    plt.show()

def toDegC(x):
    return x-273.15

if __name__ == '__main__':
    main()