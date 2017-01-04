'''
Created on Nov 7, 2013

@author: marco
'''
import os
import platform
import numpy
import pytz
import pandas as pd
import matplotlib.pyplot as plt

from estimationpy.fmu_utils.model import Model
from estimationpy.fmu_utils import csv_reader
from estimationpy.ukf.ukf_fmu import UkfFmu

import logging
from estimationpy.fmu_utils import estimationpy_logging
estimationpy_logging.configure_logger(log_level = logging.DEBUG, log_level_console = logging.INFO, log_level_file = logging.DEBUG)


def main():
    
    # Assign an existing FMU to the model, depending on the platform identified
    dir_path = os.path.dirname(__file__)
    
    # Define the path of the FMU file
    if platform.architecture()[0]=="32bit":
        print "32-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_Windows.fmu")
    else:
        print "64-bit architecture"
        filePath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_Windows.fmu")
        
    # Initialize the FMU model empty
    m = Model(filePath)
    
    # Path of the csv file containing the data series
    csvPath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "data", "NoisySimulationData_FirstOrder.csv")
    
    # Set the CSV file associated to the input, and its covariance
    input_u = m.get_input_by_name("u")
    input_u.get_csv_reader().open_csv(csvPath)
    input_u.get_csv_reader().set_selected_column("system.u")
    input_u.set_covariance(2.0)
    
    # Set the CSV file associated to the output, and its covariance
    output = m.get_output_by_name("y")
    output.get_csv_reader().open_csv(csvPath)
    output.get_csv_reader().set_selected_column("system.y")
    output.set_measured_output()
    output.set_covariance(2.0)
    
    # Select the states to be identified, and add it to the list
    m.add_variable(m.get_variable_object("x"))
    
    # Set initial value of state, and its covariance and the limits (if any)
    var = m.get_variables()[0]
    var.set_initial_value(1.5)
    var.set_covariance(0.5)
    var.set_min_value(0.0)
    var.set_constraint_low(True)
    
    # show the info about the variable to be estimated
    print var.info()
    
    # Set parameters been identified
    par_a = m.get_variable_object("a")
    m.set_real(par_a, -0.90717055)
    par_b = m.get_variable_object("b")
    m.set_real(par_b, 2.28096907)
    par_c = m.get_variable_object("c")
    m.set_real(par_c, 3.01419707)
    par_d = m.get_variable_object("d")
    m.set_real(par_d, 0.06112703)
    
    # Initialize the model for the simulation
    m.initialize_simulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = UkfFmu(m, n_proc=1)
    
    # Start the filter
    t0 = pd.to_datetime(0.0, unit = "s", utc = True)
    t1 = pd.to_datetime(30.0, unit = "s", utc = True)
    time, x, sqrtP, y, Sy, y_full = ukf_FMU.filter(start = t0, stop = t1)
    
    # Path of the csv file containing the True data series
    csvTrue = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")
    
    # Get the measured outputs
    show_results(time, x, sqrtP, y, Sy, y_full, csvTrue, csvPath, m)

def show_results(time, x, sqrtP, y, Sy, y_full, csvTrue, csvNoisy, m):
    # convert results
    x = numpy.array(x)
    y = numpy.array(y)
    sqrtP = numpy.array(sqrtP)
    Sy = numpy.array(Sy)
    y_full = numpy.squeeze(numpy.array(y_full))
    
    # Read from file
    simResults = csv_reader.CsvReader()
    simResults.open_csv(csvNoisy)
    
    simResults.set_selected_column("system.x")
    res = simResults.get_data_series()
    t_n = res.index
    d_x_n = res.values
    
    simResults = csv_reader.CsvReader()
    simResults.open_csv(csvTrue)
    
    simResults.set_selected_column("system.x")
    res = simResults.get_data_series()
    t = res.index
    d_x = res.values
    
    simResults.set_selected_column("system.y")
    res = simResults.get_data_series()
    d_y = res.values
    
    simResults.set_selected_column("system.u")
    res = simResults.get_data_series()
    d_u = res.values
    
    output = m.get_output_by_name("y")
    output.get_csv_reader().set_selected_column("system.y")
    res = output.get_csv_reader().get_data_series()
    d_y_n = res.values
    
    input_u = m.get_input_by_name("u")
    input_u.get_csv_reader().set_selected_column("system.u")
    res = input_u.get_csv_reader().get_data_series()
    d_u_n = res.values
    
    fig0 = plt.figure()
    fig0.set_size_inches(12,8)
    ax0  = fig0.add_subplot(111)
    ax0.plot(t,d_x,'g',label='$x(t)$',alpha=1.0)
    ax0.plot(time, x,'r',label='$\hat{x}(t)$')
    ax0.fill_between(time, x[:,0] - sqrtP[:,0,0], x[:,0] + sqrtP[:,0,0], facecolor='red', interpolate=True, alpha=0.3)
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('State Variable')
    ax0.set_xlim([t[0], t[-1]])
    legend = ax0.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax0.grid(False)
    plt.savefig('FirstOrder_State.png',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    fig1 = plt.figure()
    fig1.set_size_inches(12,8)
    ax1  = fig1.add_subplot(212)
    ax1.plot(t_n,d_y_n,'go',label='$y_m(t)$',alpha=1.0)
    ax1.plot(t,d_y,'g',label='$y(t)$',alpha=1.0)
    ax1.plot(time, y,'r',label='$\hat{y}(t)$')
    ax1.fill_between(time, y[:,0] - Sy[:,0,0], y[:,0] + Sy[:,0,0], facecolor='red', interpolate=True, alpha=0.3)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Output Variable')
    ax1.set_xlim([t[0], t[-1]])
    legend = ax1.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    
    ax2  = fig1.add_subplot(211)
    ax2.plot(t_n,d_u_n,'bo',label='$u_m(t)$',alpha=1.0)
    ax2.plot(t,d_u,'b',label='$u(t)$',alpha=1.0)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Input Variable')
    ax2.set_xlim([t[0], t[-1]])
    legend = ax2.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    plt.savefig('FirstOrder_InputOutput.png',dpi=400, bbox_inches='tight', transparent=True,pad_inches=0.1)
    plt.show()
    
if __name__ == '__main__':
    main()