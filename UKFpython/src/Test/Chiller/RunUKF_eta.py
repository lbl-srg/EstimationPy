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
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/ChillerFDD.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath)
    
    # Description of the model
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Path of the csv file containing the data series
    csvPath = "./ChillerResults7.csv"
    
    input = m.GetInputByName("m_flow_CW")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("m_flow_CW")
    
    input = m.GetInputByName("m_flow_CH")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("m_flow_CH")
    
    input = m.GetInputByName("T_CW_in")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("T_CW_in")
    
    input = m.GetInputByName("T_CH_in")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("T_CH_in")
    
    input = m.GetInputByName("Pin")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("P")
    
    # Set the CSV file associated to the output, and its covariance
    
    output = m.GetOutputByName("T_CH_Lea")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("T_CH_Lea")
    output.SetMeasuredOutput()
    output.SetCovariance(0.5)
    
    output = m.GetOutputByName("T_CW_lea")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("T_CW_lea")
    output.SetMeasuredOutput()
    output.SetCovariance(0.5)
    
    # Select the states to be identified, and add it to the list
    m.AddVariable(m.GetVariableObject("chi.vol1.dynBal.medium.T"))
    m.AddVariable(m.GetVariableObject("chi.vol2.dynBal.medium.T"))
    
    # Set initial value of state, and its covariance and the limits (if any)
    var = m.GetVariables()[0]
    var.SetInitialValue(300)
    var.SetCovariance(0.5)
    var.SetMinValue(273.15)
    var.SetConstraintLow(True)
    
    var = m.GetVariables()[1]
    var.SetInitialValue(279)
    var.SetCovariance(0.5)
    var.SetMinValue(273.15)
    var.SetConstraintLow(True)
    
    # Select the states to be identified, and add it to the list
    m.AddParameter(m.GetVariableObject("eta_PL"))
    
    # Set initial value of state, and its covariance and the limits (if any)
    par = m.GetParameters()[0]
    par.SetInitialValue(0.5)
    par.SetCovariance(0.05)
    par.SetMinValue(0.0)
    par.SetMaxValue(1.2)
    par.SetConstraintLow(True)
    par.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # Allow to identify eta PL externally
    m.SetReal(m.GetVariableObject("chi.external_etaPL"), True)
    m.SetReal(m.GetVariableObject("chi.external_COP"), False)
    
    # Change the nominal power of the compressor
    m.SetReal(m.GetVariableObject("P_nominal"), 1500e3)
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    
    # start filter
    time, x, sqrtP, y, Sy, y_full = ukf_FMU.filter(start = 0, stop=3600*12, verbose=False)
    
    # Get the measured outputs
    showResults(time, x, sqrtP, y, Sy, y_full, csvPath, m)

def showResults(time, x, sqrtP, y, Sy, y_full, csvTrue, m):
    
    # convert results
    x = numpy.array(x)
    y = numpy.squeeze(numpy.array(y))
    sqrtP = numpy.array(sqrtP)
    Sy = numpy.array(Sy)
    y_full = numpy.squeeze(numpy.array(y_full))
    
    print y_full
    
    # Read from file
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvTrue)
    
    simResults.SetSelectedColumn("chi.etaPL")
    res = simResults.GetDataSeries()
    t = res["time"]
    eta_PL = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("P")
    res = simResults.GetDataSeries()
    P = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("T_CH_Lea")
    res = simResults.GetDataSeries()
    T_ch = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("T_CW_lea")
    res = simResults.GetDataSeries()
    T_cw = numpy.squeeze(numpy.asarray(res["data"]))
    
    simResults.SetSelectedColumn("chi.COP")
    res = simResults.GetDataSeries()
    COP = numpy.squeeze(numpy.asarray(res["data"]))
    
    
    fig0 = plt.figure()
    fig0.set_size_inches(12,8)
    ix = 2
    ax0  = fig0.add_subplot(111)
    ax0.plot(t, eta_PL,'g',label='$\eta_{PL}$',alpha=1.0)
    ax0.plot(time, x[:,ix],'r',label='$\hat{\eta}_{PL}$')
    ax0.fill_between(time, x[:,ix] - sqrtP[:,ix,ix], x[:,ix] + sqrtP[:,ix,ix], facecolor='red', interpolate=True, alpha=0.3)
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('COP')
    ax0.set_xlim([t[0], t[-1]])
    legend = ax0.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax0.grid(False)
    plt.savefig('PartLoadEffectiveness.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    
    fig1 = plt.figure()
    ax2  = fig1.add_subplot(111)
    ax2.plot(t, T_ch,'bo',label='$T_{CH}$',alpha=1.0)
    ax2.plot(t, T_cw,'ro',label='$T_{CW}$',alpha=1.0)
    ax2.plot(time, x[:,0],'r',label='$Tcw$',alpha=1.0)
    ax2.fill_between(time, x[:,0] - sqrtP[:,0,0], x[:,0] + sqrtP[:,0,0], facecolor='red', interpolate=True, alpha=0.3)
    ax2.plot(time, x[:,1],'b',label='$Tch$',alpha=1.0)
    ax2.fill_between(time, x[:,1] - sqrtP[:,1,1], x[:,1] + sqrtP[:,1,1], facecolor='blue', interpolate=True, alpha=0.3)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Output temperatures')
    ax2.set_xlim([t[0], t[-1]])
    legend = ax2.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    plt.savefig('OutputTemperatures.pdf',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    fig2 = plt.figure()
    ax3  = fig2.add_subplot(111)
    ax3.plot(t,P,'go',label='$P$',alpha=1.0)
    ax3.plot(time, y_full[:,0],'r',label='$\hat{P}$')
    ax3.fill_between(time, y_full[:,0] - Sy[:,0,0], y_full[:,0] + Sy[:,0,0], facecolor='red', interpolate=True, alpha=0.3)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Output Variable')
    ax3.set_xlim([t[0], t[-1]])
    legend = ax3.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax3.grid(False)
    
    fig3 = plt.figure()
    ax4  = fig3.add_subplot(111)
    ax4.plot(t,COP,'k--',label='$COP$',alpha=1.0)
    ax4.plot(time, y_full[:,1],'r',label='$\hat{COP}$')
    ax4.fill_between(time, y_full[:,1] - Sy[:,1,1], y_full[:,1] + Sy[:,1,1], facecolor='red', interpolate=True, alpha=0.3)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('COP')
    ax4.set_xlim([t[0], t[-1]])
    legend = ax4.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax4.grid(False)
    
    plt.show()
    
if __name__ == '__main__':
    main()