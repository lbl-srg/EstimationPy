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
    filePath = "../../modelica/FmuExamples/Resources/FMUs/Pump_MBL3.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-4, rtol=1e-3)
    
    # Path of the csv file containing the data series
    csvPath = "../../modelica/FmuExamples/Resources/data/DataPumpVeryShort.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("Nrpm")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("Pump.Speed")
    
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("P_el")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("Pump.kW")
    output.SetMeasuredOutput()
    output.SetCovariance(0.15)
    
    """
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("V_flow")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("Pump.gpm")
    output.SetMeasuredOutput()
    output.SetCovariance(50.0)
    """
    
    #################################################################
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("pump.power.P[1]"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[0]
    par.SetInitialValue(0.5)
    par.SetCovariance(0.01)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("pump.power.P[2]"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[1]
    par.SetInitialValue(0.5)
    par.SetCovariance(0.01)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("pump.power.P[3]"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[2]
    par.SetInitialValue(0.8)
    par.SetCovariance(0.01)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("pump.power.P[4]"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[3]
    par.SetInitialValue(0.9)
    par.SetCovariance(0.01)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    ukf_FMU.setUKFparams()
    
    # start filter
    time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth = ukf_FMU.filterAndSmooth(0.0, 5.0, verbose=False)
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth, csvPath)

def showResults(time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth, csvTrue):
    # Convert list to arrays
    time = time/3600.0
    x = numpy.squeeze(numpy.array(x))
    y = numpy.squeeze(numpy.array(y))
    sqrtP = numpy.squeeze(numpy.array(sqrtP))
    Sy = numpy.squeeze(numpy.array(Sy))
    
    xs = numpy.array(Xsmooth)
    Ss = numpy.array(Ssmooth)
    Ys = numpy.array(Yfull_smooth)
    
    ####################################################################
    # Display results
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    simResults.SetSelectedColumn("Pump.kW")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    t = t/3600.0
    d_kW = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("Pump.Speed")
    res = simResults.GetDataSeries()
    d_rpm = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("Pump.gpm")
    res = simResults.GetDataSeries()
    d_gpm = numpy.squeeze(numpy.asarray(res["data"]))
    
    fig0 = plt.figure()
    fig0.set_size_inches(12,8)
    ax0  = fig0.add_subplot(111)
    #ax0.plot(time,x,'r',label='$a_5$',alpha=1.0)
    #ax0.fill_between(time, x - sqrtP, x + sqrtP, facecolor='red', interpolate=True, alpha=0.3)
    idx = 0
    ax0.plot(time,x[:,idx],'r',label='$a_1$',alpha=1.0)
    ax0.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='red', interpolate=True, alpha=0.3)
    ax0.plot(time,xs[:,idx],'r--',label='$a_1$',alpha=1.0)
    ax0.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='red', interpolate=True, alpha=0.3)
    idx = 1
    ax0.plot(time,x[:,idx],'b',label='$a_3$',alpha=1.0)
    ax0.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='blue', interpolate=True, alpha=0.3)
    ax0.plot(time,xs[:,idx],'b--',label='$a_3$',alpha=1.0)
    ax0.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='blue', interpolate=True, alpha=0.3)
    idx = 2
    ax0.plot(time,x[:,idx],'k',label='$a_5$',alpha=1.0)
    ax0.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='black', interpolate=True, alpha=0.3)
    ax0.plot(time,xs[:,idx],'k--',label='$a_5$',alpha=1.0)
    ax0.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='black', interpolate=True, alpha=0.3)
    idx = 3
    ax0.plot(time,x[:,idx],'c',label='$a_7$',alpha=1.0)
    ax0.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='cyan', interpolate=True, alpha=0.3)
    ax0.plot(time,xs[:,idx],'c--',label='$a_7$',alpha=1.0)
    ax0.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='cyan', interpolate=True, alpha=0.3)
    
    ax0.set_xlabel('Time [hours]')
    ax0.set_ylabel('Coefficients [$\cdot$]')
    ax0.set_xlim([time[0], time[-1]])
    ax0.set_ylim([0.0, 1.0])
    legend = ax0.legend(loc='upper center',bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax0.grid(False)
    
    plt.savefig('Coefficients.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results
    
    fig1 = plt.figure()
    fig1.set_size_inches(20,8)
    ax1  = fig1.add_subplot(111)
    ax1.plot(t,d_kW,'g',label='$P_{EL}^{Measured}$',alpha=1.0)
    #ax1.plot(t,d_gpm,'g',label='$\dot{m}_{PUMP}^{Measured}$',alpha=1.0)
    ax1.plot(time,y,'r',label='$P_{EL}^{UKF}$',alpha=1.0)
    ax1.plot(time,Ys[:,0],'b',label='$P_{EL}^{Smooth}$',alpha=1.0)
    #ax1.plot(time,y,'r',label='$\dot{m}_{PUMP}^{UKF}$',alpha=1.0)
    ax1.set_xlabel('Time [[hours]]')
    ax1.set_ylabel('Electrical power [kW]')
    #ax1.set_ylabel('Volume flow rate [gpm]')
    ax1.set_xlim([t[0], t[-1]])
    ax1.set_ylim([0, 4.6])
    #ax1.set_ylim([300, 600])
    legend = ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    plt.savefig('Power.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    
    plt.show()
    
if __name__ == '__main__':
    main()