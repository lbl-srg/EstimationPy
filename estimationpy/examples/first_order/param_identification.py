'''
Created on Nov 7, 2013

@author: marco
'''
import os
import platform

import numpy
import pandas as pd
import matplotlib.pyplot as plt

from estimationpy.fmu_utils.model import Model
from estimationpy.ukf.ukf_fmu import UkfFmu

def main():
    
    # Assign an existing FMU to the model, depending on the platform identified
    dir_path = os.path.dirname(__file__)
    
    # Define the path of the FMU file
    if platform.architecture()[0]=="32bit":
        print "32-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder.fmu")
    else:
        print "64-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_64bit.fmu")
    
    # Initialize the FMU model empty
    m = Model(filePath, atol=1e-6, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = os.path.join(dir_path, "..", "..", "..", "modelica", "FmuExamples", "Resources", "data", "NoisySimulationData_FirstOrder.csv")
    
    # Set the CSV file associated to the input, and its covariance
    input = m.get_input_by_name("u")
    input.get_csv_reader().open_csv(csvPath)
    input.get_csv_reader().set_selected_column("system.u")
    
    # Set the CSV file associated to the output, and its covariance
    output = m.get_output_by_name("y")
    output.get_csv_reader().open_csv(csvPath)
    output.get_csv_reader().set_selected_column("system.y")
    output.set_measured_output()
    output.set_covariance(0.5)
    
    # Set the CSV file associated to the output, and its covariance
    output = m.get_output_by_name("x")
    output.get_csv_reader().open_csv(csvPath)
    output.get_csv_reader().set_selected_column("system.x")
    output.set_measured_output()
    output.set_covariance(0.5)
    
    #################################################################
    # Select the state to be identified
    m.add_variable(m.get_variable_object("x"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.get_variables()[0]
    par.set_initial_value(1.8)
    par.set_covariance(0.5)
    
    #################################################################
    # Select the parameter to be identified
    m.add_parameter(m.get_variable_object("a"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.get_parameters()[0]
    par.set_initial_value(-0.5)
    par.set_covariance(0.01)
    par.set_min_value(-10.0)
    par.set_constraint_low(True)
    par.set_max_value(-0.1)
    par.set_constraint_high(True)
    
    
    # Select the parameter to be identified
    m.add_parameter(m.get_variable_object("b"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.get_parameters()[1]
    par.set_initial_value(0.5)
    par.set_covariance(0.01)
    par.set_min_value(0.0)
    par.set_constraint_low(True)
    par.set_max_value(20.0)
    par.set_constraint_high(True)
    
    # Select the parameter to be identified
    m.add_parameter(m.get_variable_object("c"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.get_parameters()[2]
    par.set_initial_value(0.5)
    par.set_covariance(0.01)
    par.set_min_value(0.0)
    par.set_constraint_low(True)
    par.set_max_value(20.0)
    par.set_constraint_high(True)
    
    # Select the parameter to be identified
    m.add_parameter(m.get_variable_object("d"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.get_parameters()[3]
    par.set_initial_value(0.5)
    par.set_covariance(0.01)
    par.set_min_value(0.0)
    par.set_constraint_low(True)
    par.set_max_value(20.0)
    par.set_constraint_high(True)
    
    # Initialize the model for the simulation
    m.initialize_simulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = UkfFmu(m, augmented = False)
    ukf_FMU.set_ukf_params(0.05, 2, 1)

    t0 = pd.to_datetime(0.0, unit = "s")
    t1 = pd.to_datetime(10.0, unit = "s")
    pars = ukf_FMU.parameter_estimation(t0, t1, maxIter = 600)
    print pars
    
    show_results(pars)
      
def show_results(pars):
    
    pars = numpy.squeeze(numpy.array(pars))
    
    fig0 = plt.figure()
    fig0.set_size_inches(12,8)
    ax0  = fig0.add_subplot(111)
    ax0.plot(pars, label='$pars$',alpha=1.0)
    ax0.plot([0,500],[-1,-1])
    ax0.plot([0,500],[2.5,2.5])
    ax0.plot([0,500],[3,3])
    ax0.plot([0,500],[0.1,0.1])
    ax0.axis([0,600,-2,4])
    plt.show()
    # [-1, 2.5, 3, 0.1]
    # found after 400 [-0.90717055  2.28096907  3.01419707  0.06112703]
    
    
if __name__ == '__main__':
    main()