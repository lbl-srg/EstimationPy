'''
Created on Nov 7, 2013

@author: marco
'''
import os
import platform

import matplotlib.pyplot as plt
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
    
    # ReInit the model with the new FMU
    m.re_init(filePath)
    
    # Show details
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.get_input_names(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.get_output_names(), "\n"
    
    # Set the CSV file associated to the input
    inputPath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")
    input_u = m.get_input_by_name("u")
    input_u.get_csv_reader().open_csv(inputPath)
    input_u.get_csv_reader().set_selected_column("system.u")
    
    # Initialize the model for the simulation
    m.initialize_simulator()
                      
    # Simulate
    time, results = m.simulate()
    
    # Show the results
    show_results(time, results)
    
def show_results(time, results):
    
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(211)
    ax1.plot(time,results["x"],'g',label='$x(t)$',alpha=1.0)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('State variable')
    ax1.set_xlim([time[0], time[-1]])
    legend = ax1.legend(loc='upper left', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    
    ax2  = fig1.add_subplot(212)
    ax2.plot(time,results["y"],'b',label='$y(t)$',alpha=1.0)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Output variable')
    ax2.set_xlim([time[0], time[-1]])
    legend = ax2.legend(loc='upper left', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    plt.savefig('FirstOrder.png',dpi=300, bbox_inches='tight', transparent=True,pad_inches=0.1)
    plt.show()
   
if __name__ == '__main__':
    main()