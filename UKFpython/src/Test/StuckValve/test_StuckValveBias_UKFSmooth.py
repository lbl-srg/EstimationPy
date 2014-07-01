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
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/Fmu_ValveStuck_bias3.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-5, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = "../../../modelica/FmuExamples/Resources/data/NoisyData_ValveBias4.csv"
    
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
    m.AddVariable(m.GetVariableObject("command.y"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    var = m.GetVariables()[0]
    var.SetInitialValue(0.8)
    var.SetCovariance(0.05)
    var.SetMinValue(0.0)
    var.SetConstraintLow(True)
    var.SetMaxValue(1.00)
    var.SetConstraintHigh(True)
    
    #################################################################
    # Select the variable to be estimated
    m.AddParameter(m.GetVariableObject("lambda"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    var = m.GetParameters()[0]
    var.SetInitialValue(0.00)
    var.SetCovariance(0.0007)
    var.SetMinValue(-0.005)
    var.SetConstraintLow(True)
    var.SetMaxValue(0.025)
    var.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # Set a parameter of the model
    # This parameter specifies
    use_cmd = m.GetVariableObject("use_cmd")
    m.SetReal(use_cmd, 0.0)
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    ukf_FMU.setUKFparams(alpha=0.001, k=0)
    
    # start filter
    time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth = ukf_FMU.filterAndSmooth(0.0, 5.0, verbose=False)
    
    # Path of the csv file containing the True data series
    csvTrue = "../../../modelica/FmuExamples/Resources/data/SimulationData_ValveBias4.csv"
    
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
    simResults.SetSelectedColumn("valveStuck.T_in")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    d_temp = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("valveStuck.dp")
    res = simResults.GetDataSeries()
    
    d_dp = numpy.squeeze(numpy.asarray(res["data"]))
    
    input = m.GetInputByName("T_in")
    input.GetCsvReader().SetSelectedColumn("valveStuck.T_in")
    res = input.GetCsvReader().GetDataSeries()
    
    t_t = res["time"]
    d_temp_noisy = numpy.squeeze(numpy.asarray(res["data"]))
    
    input = m.GetInputByName("dp")
    input.GetCsvReader().SetSelectedColumn("valveStuck.dp")
    res = input.GetCsvReader().GetDataSeries()
    
    d_dp_noisy = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("valveStuck.m_flow_real")
    res = simResults.GetDataSeries()
    
    d_real = numpy.squeeze(numpy.asarray(res["data"]))
    
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
    #ax0.plot(time,y,'r',label='$\dot{m}^{Filter}$')
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
    
    simResults.SetSelectedColumn("valveStuck.valve.opening")
    res = simResults.GetDataSeries()
    opening = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("valveStuck.cmd")
    res = simResults.GetDataSeries()
    command = numpy.squeeze(numpy.asarray(res["data"]))
    
    
    fig1 = plt.figure()
    idx = 0
    fig1.set_size_inches(12,8)
    ax1  = fig1.add_subplot(111)
    ax1.plot(t,command,'g',label='$u$',alpha=1.0)
    ax1.plot(t,opening,'k',label='$x$',alpha=1.0)
    ax1.plot(time,x[:,idx],'r',label='$\hat{x}$')
    ax1.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='red', interpolate=True, alpha=0.3)
    ax1.plot(time,xs[:,idx],'b',label='$\hat{x}^S$')
    ax1.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='blue', interpolate=True, alpha=0.3)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Valve opening [$\cdot$]')
    ax1.set_xlim([t[0], t[-1]])
    ax1.set_ylim([0, 1.1])
    legend = ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    plt.savefig('Positions.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    fig2 = plt.figure()
    idx = 0
    fig2.set_size_inches(12,8)
    ax2  = fig2.add_subplot(211)
    ax2.plot(t,toDegC(d_temp),'b',label='$T$',alpha=1.0)
    ax2.plot(t_t,toDegC(d_temp_noisy),'bo',label='$T^{N}$',alpha=0.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Water temperature [$^{\circ}$C]')
    ax2.set_xlim([t[0], t[-1]])
    ax2.set_ylim([toDegC(273.15+14), toDegC(273.15+50)])
    legend = ax2.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    plt.savefig('Temperature.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ax4  = fig2.add_subplot(212)
    ax4.plot(t,d_dp/1e5,'g',label='$\Delta p$',alpha=1.0)
    ax4.plot(t_t,d_dp_noisy/1e5,'go',label='$\Delta p^{N}$',alpha=0.5)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Pressure difference [$bar$]')
    ax4.set_xlim([t[0], t[-1]])
    ax4.set_ylim([0, 0.6])
    legend = ax4.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax4.grid(False)
    plt.savefig('Pressure.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    fig3 = plt.figure()
    idx = 1
    fig3.set_size_inches(12,8)
    ax3  = fig3.add_subplot(111)
    ax3.plot(t,d_lambda,'g',label='$\lambda$')
    ax3.plot(time,x[:,idx],'r',label='$\hat{\lambda}$')
    ax3.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='red', interpolate=True, alpha=0.3)
    ax3.plot(time,xs[:,idx],'b',label='$\hat{\lambda}^S$')
    ax3.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='blue', interpolate=True, alpha=0.3)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Sensor thermal drift coeff. [$1/K$]')
    ax3.set_xlim([t[0], t[-1]])
    ax3.set_ylim([-0.0025, 0.015])
    legend = ax3.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax3.grid(False)
    plt.savefig('Drift.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    plt.show()

def toDegC(x):
    return x-273.15

if __name__ == '__main__':
    main()