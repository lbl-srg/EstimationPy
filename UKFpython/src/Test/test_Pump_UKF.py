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
    filePath = "../../modelica/FmuExamples/Resources/FMUs/Pump.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-3, rtol=1e-3)
    
    # Path of the csv file containing the data series
    csvPath = "../../modelica/FmuExamples/Resources/data/DataPump2.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("Nrpm")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("Pump.Speed")
    
    """
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("P_el")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("Pump.kW")
    output.SetMeasuredOutput()
    output.SetCovariance(0.5)
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
    """
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("a"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[0]
    par.SetInitialValue(0.0)
    par.SetCovariance(0.1)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("a1"))
    
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
    # Convert list to arrays
    time = time/3600.0
    x = numpy.squeeze(numpy.array(x))
    y = numpy.squeeze(numpy.array(y))
    sqrtP = numpy.squeeze(numpy.array(sqrtP))
    Sy = numpy.squeeze(numpy.array(Sy))
    
    print y
    print numpy.shape(y)
    
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
    ax0  = fig0.add_subplot(211)
    ax0.plot(time,x[:,0],'k',label='$a_1$',alpha=1.0)
    ax0.plot(time,x[:,1],'b',label='$a_2$',alpha=1.0)
    ax0.set_xlabel('Time [hours]')
    ax0.set_ylabel('Coefficients [$\cdot$]')
    ax0.set_xlim([time[0], time[-1]])
    ax0.set_ylim([-0.1, 3])
    legend = ax0.legend(loc='upper center',bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax0.grid(False)
    
    """
    ax0a  = fig0.add_subplot(212)
    ax0a.plot(time,x[:,2],'k',label='$b_1$',alpha=1.0)
    ax0a.plot(time,x[:,3],'b',label='$b_2$',alpha=1.0)
    ax0a.set_xlabel('Time [hours]')
    ax0a.set_ylabel('Coefficients [$\cdot$]')
    ax0a.set_xlim([time[0], time[-1]])
    ax0a.set_ylim([0, 3])
    legend = ax0a.legend(loc='upper center',bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax0a.grid(False)
    """
    
    plt.savefig('Coefficients.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results
    
    fig1 = plt.figure()
    fig1.set_size_inches(20,8)
    ax1  = fig1.add_subplot(111)
    #ax1.plot(t,d_kW,'g',label='$P_{EL}^{Measured}$',alpha=1.0)
    ax1.plot(t,d_gpm,'g',label='$\dot{m}_{PUMP}^{Measured}$',alpha=1.0)
    #ax1.plot(time,y,'r',label='$P_{EL}^{UKF}$',alpha=1.0)
    ax1.plot(time,y,'r',label='$\dot{m}_{PUMP}^{UKF}$',alpha=1.0)
    ax1.set_xlabel('Time [[hours]]')
    #ax1.set_ylabel('Electrical power [kW]')
    ax1.set_ylabel('Volume flow rate [gpm]')
    ax1.set_xlim([t[0], t[-1]])
    #ax1.set_ylim([0, 4.6])
    ax1.set_ylim([300, 600])
    legend = ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    plt.savefig('Power.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results
    
    a = x[-1][0]
    a1 = x[-1][1]
    
    a_av = numpy.average(x[:][0])
    a1_av = numpy.average(x[:][1])
    
    print a, a1
    print a_av, a1_av
    
    rpm = numpy.linspace(35.0,60.0,100)
    x_n = rpm/60.0
    #app = 4.4*(a*x_n*x_n + (1 - a)*x_n)**a1
    #app_av = 4.4*(a_av*x_n*x_n + (1 - a_av)*x_n)**a1_av
    
    app = 600*(a*x_n*x_n + (1 - a)*x_n)**a1
    app_av = 600*(a_av*x_n*x_n + (1 - a_av)*x_n)**a1_av
    
    fig2 = plt.figure()
    fig2.set_size_inches(10,10)
    ax2  = fig2.add_subplot(111)
    #ax2.plot(d_rpm,d_kW,'bo',label='$P_{EL}^{Measured}$',alpha=1.0)
    ax2.plot(d_rpm,d_gpm,'bo',label='$\dot{m}_{PUMP}^{Measured}$',alpha=1.0)
    #ax2.plot(rpm, app,'r',label='$P_{EL}^{Model}$',alpha=1.0)
    ax2.plot(rpm, app,'r',label='$\dot{m}_{PUMP}^{Model}$',alpha=1.0)
    ax2.set_xlabel('Motor speed [rpm]')
    #ax2.set_ylabel('Electrical power [kW]')
    ax2.set_ylabel('Volume flow rate [gpm]')
    ax2.set_xlim([rpm[0], rpm[-1]])
    ax2.set_ylim([300, 600])
    legend = ax2.legend(loc='upper left',bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    plt.savefig('Curve.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results
    
    
    
    plt.show()
    
if __name__ == '__main__':
    main()