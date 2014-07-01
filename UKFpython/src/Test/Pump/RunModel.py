'''
Created on Nov 7, 2013

@author: marco
'''
import numpy
import matplotlib.pyplot as plt

from FmuUtils import Model
from FmuUtils import CsvReader

def main():
    
    # Initialize the FMU model empty
    m = Model.Model()
    
    # Assign an existing FMU to the model
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/Pump_MBL3.fmu"
    
    # ReInit the model with the new FMU
    m.ReInit(filePath)
    
    # Show details
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Path of the csv file containing the data series
    #csvPath = "../../../modelica/FmuExamples/Resources/data/DataPumpVeryShort.csv"
    csvPath = "../../../modelica/FmuExamples/Resources/data/DataPump_16to19_Oct2012.csv"
    #csvPath = "../../../modelica/FmuExamples/Resources/data/DataPump_16to19_Oct2012_variableStep.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("Nrpm")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("Pump.Speed")
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    #pars = [0.106,  0.50041746, 0.9, 1.]
    #pars = [0.11574544, 0.67699191, 0.8862938, 1.]
    pars = [ 0.11992406,  0.11864333,  0.88393507,  1.]
    #pars = [0.1222095,  0.1210177,  0.12177384, 1.]
    
    # Set the parameters of the model
    par_1 = m.GetVariableObject("pump.power.P[1]")
    m.SetReal(par_1, pars[0])
    
    par_2 = m.GetVariableObject("pump.power.P[2]")
    m.SetReal(par_2, pars[1])
    
    par_3 = m.GetVariableObject("pump.power.P[3]")
    m.SetReal(par_3, pars[2])
    
    par_4 = m.GetVariableObject("pump.power.P[4]")
    m.SetReal(par_4, pars[3])
                     
    # Simulate
    time, results = m.Simulate()
    
    # Show the results
    showResults(time, results, csvPath, pars)
    
def showResults(time, results, csvPath, pars):
    
    ####################################################################
    # Display results
    simResults = CsvReader.CsvReader()
    simResults.OpenCSV(csvPath)
    simResults.SetSelectedColumn("Pump.kW")
    res = simResults.GetDataSeries()
    
    t = res["time"]
    t = t/3600.0
    time = time/3600
    d_kW = numpy.squeeze(numpy.asarray(res["data"]))
    P_el = results["P_el"]
    
    Ndata = numpy.max([len(d_kW), len(P_el)])
    new_t = numpy.linspace(time[0], time[-1], Ndata)
    d_kW = numpy.interp(new_t, t, d_kW)
    P_el = numpy.interp(new_t, time, P_el)
    
    error = d_kW - P_el
    MSE   = numpy.sqrt(numpy.sum(numpy.power(error, 2.0)))
    
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(111)
    ax1.plot(new_t, P_el,'r',label='$P_{kW}^{MODEL}$',alpha=1.0)
    ax1.plot(new_t, d_kW,'g',label='$P_{kW}^{DATA}$',alpha=1.0)
    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('Power [kW]')
    ax1.set_title('$MSE=%.6f$ \n $p_1=%.4f \ p_2=%.4f \ p_3=%.4f \ p_4=%.4f$' %(MSE, pars[0], pars[1], pars[2], pars[3]))
    ax1.set_xlim([time[0], time[-1]])
    ax1.set_ylim([0, 5])
    legend = ax1.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    
    plt.savefig('PumpElectrical.pdf',dpi=300, bbox_inches='tight', transparent=True,pad_inches=0.1)
    corr_coeff = numpy.corrcoef(d_kW, P_el)
    
    fig2 = plt.figure()
    ax2  = fig2.add_subplot(111)
    ax2.plot(d_kW/numpy.max(d_kW), P_el/numpy.max(P_el), 'ro', alpha=0.5)
    ax2.plot(d_kW/numpy.max(d_kW), d_kW/numpy.max(d_kW), 'b', alpha=1.0)
    ax2.set_xlabel('Normalized measured power')
    ax2.set_ylabel('Normalized simulated power')
    ax2.set_title('$r=%.6f$' %(corr_coeff[0,1]))
    ax2.set_ylim([0, 1.1])
    ax2.set_ylim([0, 1.1])
    ax2.grid(False)
    
    plt.savefig('PumpElectrical_2.pdf',dpi=300, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    plt.show()
   
if __name__ == '__main__':
    main()