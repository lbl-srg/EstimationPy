'''
Created on Nov 7, 2013

@author: marco
'''
import numpy
import matplotlib.pyplot as plt
    
from FmuUtils.FmuPool import FmuPool
from FmuUtils.Model import Model

def main():

    # Initialize the FMU model empty
    m = Model()
    
    # Assign an existing FMU to the model
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/HeatExchanger.fmu"
    
    # ReInit the model with the new FMU
    m.ReInit(filePath, atol=1e-5, rtol=1e-6)
    
    # Set the CSV file associated to the input
    inputPath = "../../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    input = m.GetInputByName("mFlow_cold")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.mFlow_COLD")
        
    inputPath = "../../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    input = m.GetInputByName("mFlow_hot")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.mFlow_HOT")
        
    inputPath = "../../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    input = m.GetInputByName("T_hot")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.Thot_IN")
        
    inputPath = "../../../modelica/FmuExamples/Resources/data/SimulationData_HeatExchanger.csv"
    input = m.GetInputByName("T_cold")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("heatExchanger.Tcold_IN")
    
    # Select the states to be modified
    m.AddVariable(m.GetVariableObject("metal.T"))
    
    # Initialize the model for the simulation
    m.InitializeSimulator()

    # Instantiate the pool
    pool = FmuPool(m, debug = False)

    # define the vector of initial conditions for which the simulations
    # have to be performed.
    # values has to be a list of state vectors
    # values = [ [x0_0], [x0_1], ... [x0_n]]
    vectorValues = numpy.linspace(283.15, 303.15, 20)
    values = []
    for v in vectorValues:
        temp = {"state":numpy.array([v]), "parameters":[]}
        values.append(temp)
    
    # Run simulations in parallel
    poolResults = pool.Run(values, stop=200.0)
    
    # plot all the results
    showResults(poolResults)    
    

def showResults(poolResults):
    
    # Create the figure
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(111)
    
    
    for res in poolResults:
        # get the results of a worker of the pool
        time, results = res[0]
        
        # plot
        ax1.plot(time,results["Thot_IN"],'r',label='$T_{Hot}^{IN}$',alpha=0.6)
        ax1.plot(time,results["Thot_OUT"],'r--',label='$T_{Hot}^{OUT}$',alpha=0.6)
        ax1.plot(time,results["Tcold_IN"],'b',label='$T_{Cold}^{IN}$',alpha=0.6)
        ax1.plot(time,results["Tcold_OUT"],'b--',label='$T_{Cold}^{OUT}$',alpha=0.6)
        ax1.plot(time,results["Tmetal"],'k',label='$T_{Metal}$',alpha=0.6)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Water temperatures ')
    ax1.set_xlim([time[0], time[-1]])
    ax1.grid(False)
    
    plt.show()
    
if __name__ == '__main__':
    main()