'''
Created on Nov 7, 2013

@author: marco
'''
import os
import platform
import numpy
import matplotlib.pyplot as plt
    
from estimationpy.fmu_utils.fmu_pool import FmuPool
from estimationpy.fmu_utils.model import Model

def main():

    # Initialize the FMU model empty
    m = Model()

    # Assign an existing FMU to the model, depending on the platform identified
    dir_path = os.path.dirname(__file__)

    # Define the path of the FMU file
    if platform.architecture()[0]=="32bit":
        print "32-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder.fmu")
    else:
        print "64-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_64bit.fmu")

    # Assign an existing FMU to the model
    m.re_init(filePath)

    # Set the CSV file associated to the input
    inputPath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")
    input_u = m.get_input_by_name("u")
    input_u.get_csv_reader().open_csv(inputPath)
    input_u.get_csv_reader().set_selected_column("system.u")
    
    # Select the states to be modified
    m.add_variable(m.get_variable_object("x"))

    # Initialize the simulator
    m.initialize_simulator()

    # Instantiate the pool
    pool = FmuPool(m, debug = False)

    # define the vector of initial conditions for which the simulations
    # have to be performed.
    # values has to be a list of state vectors
    # values = [ [x0_0], [x0_1], ... [x0_n]]
    vector_values = numpy.linspace(1.0, 5.0, 10)
    values = []
    for v in vector_values:
        temp = {"state":numpy.array([v]), "parameters":[]}
        values.append(temp)
    
    # Run simulations in parallel
    pool_results = pool.run(values)
    
    # plot all the results
    show_results(pool_results)
    

def show_results(pool_results):
    
    # Create the figure
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(211)
    ax2  = fig1.add_subplot(212)
    
    
    for res in pool_results:
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
    plt.savefig('FirstOrderPool.png',dpi=300, bbox_inches='tight', transparent=True,pad_inches=0.1)
    plt.show()
    
if __name__ == '__main__':
    main()