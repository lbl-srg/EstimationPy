'''
Created on Nov 7, 2013

@author: marco
'''
import numpy
import matplotlib.pyplot as plt

from FmuUtils import Model
from FmuUtils.FmuPool import FmuPool

def main():
    
    # Assign an existing FMU to the model
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/Pump_MBL2.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-4, rtol=1e-4)
    
    # Show details of the model
    print m
    
    # Path of the csv file containing the data series
    # csvPath = "../../../modelica/FmuExamples/Resources/data/DataPumpShort.csv"
    csvPath = "../../../modelica/FmuExamples/Resources/data/DataPump_16to19_Oct2012.csv"
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("Nrpm")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("Pump.Speed")
    
    # Select the states to be modified
    m.AddParameter(m.GetVariableObject("pump.power.P[1]"))
    m.AddParameter(m.GetVariableObject("pump.power.P[2]"))
    m.AddParameter(m.GetVariableObject("pump.power.P[3]"))

    # Initialize the simulator
    m.InitializeSimulator()
    
    # Instantiate the pool
    pool = FmuPool(m, debug = True)

    # define the vector of initial conditions for which the simulations
    # have to be performed.
    # values has to be a list of state vectors
    # values = [ [x0_0], [x0_1], ... [x0_n]]
    vectorValues = numpy.linspace(0.2, 1.0, 5)
    values = []
    for v in vectorValues:
        temp = {"state":[], "parameters":numpy.array([v,v,v])}
        values.append(temp)
    
    # Run simulations in parallel
    poolResults = pool.Run(values)
    
    # plot all the results
    showResults(poolResults)

def showResults(poolResults):
    
    # Create the figure
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(211)
    ax2  = fig1.add_subplot(212)
    
    for res in poolResults:
        # get the results of a worker of the pool
        time, results = res[0]
        
        # plot
        ax1.plot(time,results["P_el"],'r',alpha=0.6)
        ax2.plot(time,results["V_flow"],'b',alpha=0.6) 
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Power [kW]')
    ax1.set_xlim([time[0], time[-1]])
    ax1.grid(False)
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Mass flow rate [gpm]')
    ax2.set_xlim([time[0], time[-1]])
    ax2.grid(False)
    plt.savefig('PumpPool.pdf',dpi=300, bbox_inches='tight', transparent=True,pad_inches=0.1)
    plt.show()
    
if __name__ == '__main__':
    main()