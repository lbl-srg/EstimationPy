'''
Created on Nov 7, 2013

@author: marco
'''
import os
import platform
import numpy
import matplotlib.pyplot as plt
    
from FmuUtils.FmuPool import FmuPool
from FmuUtils.Model import Model

def main():

    # Initialize the FMU model empty
    m = Model()

    # Assign an existing FMU to the model, depending on the platform identified
    dir_path = os.path.dirname(__file__)

    # Define the path of the FMU file
    if platform.architecture()[0]=="32bit":
        print "32-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder.fmu")
    else:
        print "64-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_64bit.fmu")

    # Assign an existing FMU to the model
    m.ReInit(filePath)

    # Set the CSV file associated to the input
    inputPath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")
    input_u = m.GetInputByName("u")
    input_u.GetCsvReader().OpenCSV(inputPath)
    input_u.GetCsvReader().SetSelectedColumn("system.u")
    
    # Select the states to be modified
    m.AddVariable(m.GetVariableObject("x"))

    # Initialize the simulator
    m.InitializeSimulator()

    # Instantiate the pool
    pool = FmuPool(m, debug = False)

    # define the vector of initial conditions for which the simulations
    # have to be performed.
    # values has to be a list of state vectors
    # values = [ [x0_0], [x0_1], ... [x0_n]]
    vectorValues = numpy.linspace(1.0, 5.0, 10)
    values = []
    for v in vectorValues:
        temp = {"state":numpy.array([v]), "parameters":[]}
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
        ax1.plot(time,results["x"],'g',alpha=0.6)
        ax2.plot(time,results["y"],'b',alpha=0.6) 
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('State variable')
    ax1.set_xlim([time[0], time[-1]])
    ax1.grid(False)
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Output variable')
    ax2.set_xlim([time[0], time[-1]])
    ax2.grid(False)
    plt.savefig('FirstOrderPool.pdf',dpi=300, bbox_inches='tight', transparent=True,pad_inches=0.1)
    plt.show()
    
if __name__ == '__main__':
    main()