'''
Created on Nov 7, 2013

@author: marco
'''
import os
import platform

import numpy
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from scipy.stats import norm
from pylab import figure

from estimationpy.fmu_utils.model import Model
from estimationpy.fmu_utils import csv_reader
from estimationpy.ukf.ukf_fmu import UkfFmu

import matplotlib.dates as mdates

def main():
    """
    This example demonstrates how state and parameters can be simultaneously
    estimated to identify faults in a valve.
    """
    
    # Assign an existing FMU to the model, depending on the platform identified
    dir_path = os.path.dirname(__file__)
    
    # Define the path of the FMU file
    filePath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "FMUs", "ValveStuck.fmu")
    
    # Initialize the FMU model empty
    m = Model(filePath, atol=1e-5, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "data", "NoisyData_ValveBias4.csv")
    
    # Set the CSV file associated to the input, and its covariance
    input = m.get_input_by_name("dp")
    input.get_csv_reader().open_csv(csvPath)
    input.get_csv_reader().set_selected_column("valveStuck.dp")
    
    # Set the CSV file associated to the input, and its covariance
    input = m.get_input_by_name("cmd")
    input.get_csv_reader().open_csv(csvPath)
    input.get_csv_reader().set_selected_column("valveStuck.cmd")
    
    # Set the CSV file associated to the input, and its covariance
    input = m.get_input_by_name("T_in")
    input.get_csv_reader().open_csv(csvPath)
    input.get_csv_reader().set_selected_column("valveStuck.T_in")
    
    # Set the CSV file associated to the output, and its covariance
    output = m.get_output_by_name("m_flow")
    output.get_csv_reader().open_csv(csvPath)
    output.get_csv_reader().set_selected_column("valveStuck.m_flow")
    output.set_measured_output()
    output.set_covariance(0.05)
    
    
    #################################################################
    # Select the variable to be estimated
    m.add_variable(m.get_variable_object("command.y"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    var = m.get_variables()[0]
    var.set_initial_value(1.0)
    var.set_covariance(0.05)
    var.set_min_value(0.0)
    var.set_constraint_low(True)
    var.set_max_value(1.00)
    var.set_constraint_high(True)
    
    #################################################################
    # Select the parameter to be estimated
    m.add_parameter(m.get_variable_object("lambda"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    var = m.get_parameters()[0]
    var.set_initial_value(0.00)
    var.set_covariance(0.0007)
    var.set_min_value(-0.005)
    var.set_constraint_low(True)
    var.set_max_value(0.025)
    var.set_constraint_high(True)
    
    # Initialize the model for the simulation
    m.initialize_simulator()
    
    # Set models parameters
    use_cmd = m.get_variable_object("use_cmd")
    m.set_real(use_cmd, 0.0)
    Lambda = m.get_variable_object("lambda")
    m.set_real(Lambda, 0.0)
    
    #################################################################
    # Instantiate the UKF for the FMU model
    ukf_FMU = UkfFmu(m)
    
    # Start filter
    t0 = pd.to_datetime(0.0, unit = "s", utc = True)
    t1 = pd.to_datetime(360.0, unit = "s", utc = True)
    time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth = \
        ukf_FMU.filter_and_smooth(start = t0, stop = t1, verbose = False)
    
    # Plot the results
    showResults(time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth, m)

def showResults(time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth, m):
    """
    Function that generates the plots containing the results of the example
    """
    
    # Path of the csv file containing the True data series
    dir_path = os.path.dirname(__file__)
    csvTrue = os.path.join(dir_path, "..", "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_ValveBias4.csv")
    
    ####################################################################
    # Convert the results computed by the filter and smoother
    # from lists to arrays
    x = numpy.array(x)
    y = numpy.array(y)
    sqrtP = numpy.array(sqrtP)
    Sy = numpy.array(Sy)
    y_full = numpy.squeeze(numpy.array(y_full))
    xs = numpy.array(Xsmooth)
    Ss = numpy.array(Ssmooth)
    Ys = numpy.array(Yfull_smooth)
    
    ####################################################################
    # Collect data to plot from CSV files
    simResults = csv_reader.CsvReader()
    simResults.open_csv(csvTrue)
    simResults.set_selected_column("valveStuck.T_in")
    res = simResults.get_data_series()
    t = res.index
    d_temp = res.values
    
    simResults.set_selected_column("valveStuck.dp")
    res = simResults.get_data_series()
    d_dp = res.values
    
    inp = m.get_input_by_name("T_in")
    inp.get_csv_reader().set_selected_column("valveStuck.T_in")
    res = inp.get_csv_reader().get_data_series()
    
    t_t = res.index
    d_temp_noisy = res.values
    
    inp = m.get_input_by_name("dp")
    inp.get_csv_reader().set_selected_column("valveStuck.dp")
    res = inp.get_csv_reader().get_data_series()
    d_dp_noisy = res.values
    
    simResults.set_selected_column("valveStuck.m_flow_real")
    res = simResults.get_data_series()
    d_real = res.values
    
    simResults.open_csv(csvTrue)
    simResults.set_selected_column("valveStuck.lambda")
    res = simResults.get_data_series()
    d_lambda = res.values
    
    outputRes = m.get_output_by_name("m_flow").get_csv_reader()
    outputRes.set_selected_column("valveStuck.m_flow")
    res = outputRes.get_data_series()
    
    to = res.index
    do = res.values
    
    simResults.set_selected_column("valveStuck.valve.opening")
    res = simResults.get_data_series()
    opening = res.values
    
    simResults.set_selected_column("valveStuck.cmd")
    res = simResults.get_data_series()
    command = res.values
    
    ####################################################################
    # Compute and plot fault probabilities
    probFault, faultStatus = computeProbabilities(t, command, opening, time, x, sqrtP, xs, Ss)
    
    # plot the probabilities
    fig4 = plt.figure()
    fig4.set_size_inches(10, 7)
    ax4  = fig4.add_subplot(211)
    ax4.plot(time, probFault[:,0],'k--',label='$P_{Leak}$')
    ax4.plot(time, probFault[:,1],'k',label='$P_{Stuck}$')
    ax4.set_ylabel('Probability of fault [%]')
    ax4.set_xlim([time[0], time[-1]])
    ax4.set_ylim([0, 105])
    legend = ax4.legend(loc='lower left', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax4.grid(False)
    
    ax5 = fig4.add_subplot(212)
    plotFaultStatus(time, faultStatus, ax5)
    plt.savefig('Probability.png', dpi=200, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results: mass flow rate
    fig0 = plt.figure()
    fig0.set_size_inches(10, 7)
    ax0  = fig0.add_subplot(111)
    ax0.plot(t,d_real,'g',label='$\dot{m}$',alpha=1.0)
    ax0.plot(to,do,'go',label='$\dot{m}^{N+D}$',alpha=0.5)
    ax0.plot(time,Ys[:,0],'b',label='$\hat{\dot{m}}$')
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('Mass flow rate [kg/s]')
    ax0.set_xlim([time[0], time[-1]])
    ax0.set_ylim([0, 1.3])
    ax0.fmt_xdata = mdates.DateFormatter('%M:%S')
    legend = ax0.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax0.grid(False)
    plt.savefig('Flow.png',dpi=200, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results: position of the valve
    fig1 = plt.figure()
    idx = 0
    fig1.set_size_inches(10, 7)
    gs = gridspec.GridSpec(2, 1, height_ratios=[7,3])
    ax1  = plt.subplot(gs[0])
    ax1.plot(t,command,'g',label='$u$',alpha=1.0)
    ax1.plot(t,opening,'r',label='$x$',alpha=1.0)
    ax1.plot(time,xs[:,idx],'b',label='$\hat{x}$')
    ax1.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='blue', interpolate=True, alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Valve opening [$\cdot$]')
    ax1.set_xlim([time[0], time[-1]])
    ax1.set_ylim([0, 1.1])
    ax1.fmt_xdata = mdates.DateFormatter('%M:%S')
    legend = ax1.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    
    ax1_bis  = plt.subplot(gs[1])
    ax1_bis.fmt_xdata = mdates.DateFormatter('%M:%S')
    plotFaultStatus(time, faultStatus, ax1_bis)
    plt.savefig('Positions.png',dpi=200, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    
    ####################################################################
    # Display results: temperature of the water and pressure drop
    fig2 = plt.figure()
    idx = 0
    fig2.set_size_inches(10, 7)
    ax2  = fig2.add_subplot(211)
    ax2.plot(t,toDegC(d_temp),'b',label='$T$',alpha=1.0)
    ax2.plot(t_t,toDegC(d_temp_noisy),'bo',label='$T^{N}$',alpha=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Water temperature [$^{\circ}$C]')
    ax2.set_xlim([time[0], time[-1]])
    ax2.set_ylim([toDegC(273.15+14), toDegC(273.15+50)])
    ax2.fmt_xdata = mdates.DateFormatter('%M:%S')
    legend = ax2.legend(loc='lower center', ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    ax4  = fig2.add_subplot(212)
    ax4.plot(t,d_dp/1e5,'g',label='$\Delta p$',alpha=1.0)
    ax4.plot(t_t,d_dp_noisy/1e5,'go',label='$\Delta p^{N}$',alpha=0.5)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Pressure difference [$bar$]')
    ax4.set_xlim([time[0], time[-1]])
    ax4.set_ylim([0, 0.6])
    ax4.fmt_xdata = mdates.DateFormatter('%M:%S')
    legend = ax4.legend(loc='lower center', ncol=4, fancybox=True, shadow=True)
    legend.draggable()
    ax4.grid(False)
    plt.savefig('Temp_Press.png',dpi=200, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    ####################################################################
    # Display results: estimated drift coefficient
    fig3 = plt.figure()
    idx = 1
    fig3.set_size_inches(12, 6)
    ax3  = fig3.add_subplot(111)
    ax3.plot(t,d_lambda,'g',label='$\lambda$')
    ax3.plot(time,x[:,idx],'r',label='$\hat{\lambda}$')
    ax3.fill_between(time, x[:,idx] - sqrtP[:,idx,idx], x[:,idx] + sqrtP[:,idx,idx], facecolor='red', interpolate=True, alpha=0.3)
    ax3.plot(time,xs[:,idx],'b',label='$\hat{\lambda}^S$')
    ax3.fill_between(time, xs[:,idx] - Ss[:,idx,idx], xs[:,idx] + Ss[:,idx,idx], facecolor='blue', interpolate=True, alpha=0.3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Sensor thermal drift coeff. [$1/K$]')
    ax3.set_xlim([time[0], time[-1]])
    ax3.set_ylim([-0.0025, 0.015])
    ax3.fmt_xdata = mdates.DateFormatter('%M:%S')
    legend = ax3.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax3.grid(False)
    plt.savefig('Drift.png',dpi=200, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    
    plt.show()

def toDegC(x):
    """
    This function converts from degree kelvin to Celsius
    """
    return x-273.15

def computeProbabilities(t, command, opening, time, x, sqrtP, Xsmooth, Ssmooth):
    """
    This function takes the results of the UKF and of the smoother, computes
    the fault probabilities and identifies the periods when the faults
    are likely to happen.
    """
    import time as T
    
    # COMPUTE FAULT PROBABILITIES
    # Define the thresholds
    Thresh  = 0.075
    
    numSamples = len(time)
    
    realFault   = numpy.zeros((numSamples,2))
    faultStatus = numpy.zeros((numSamples,2))
    probFault   = numpy.zeros((numSamples,2))
    
    # converting the squared matrix into the covariance ones
    Psmooth   = numpy.zeros(Ssmooth.shape)
    (N, I, J) = Ssmooth.shape
    for n in range(N):
        Psmooth[n,:,:] = numpy.dot(Ssmooth[n,:,:], Ssmooth[n,:,:].T)
    
    # interpolate the command to have the same dimension of the estimations
    new_t = numpy.array([T.mktime(x.timetuple()) for x in t.tolist() ])
    new_time = numpy.array( [T.mktime(x.timetuple()) for x in time.tolist() ])
    command = numpy.interp(new_time,  new_t, command)
    opening = numpy.interp(new_time,  new_t, opening)
    
    for i in range(numSamples):
        
        # ComputingFault probabilities with smoothed estimation
        StdDev = numpy.diag([Ssmooth[i,0,0]])
        
        errorLEAK = numpy.array([Xsmooth[i,0] - (command[i]-Thresh)])
        errorLEAK = errorLEAK/numpy.diag(StdDev)
        probFault[i,0] = 100.0*norm.cdf(errorLEAK)
        
        faultStatus[i,0] = 0.48 if Xsmooth[i,0] > (command[i]+Thresh) else 0.02
        
        errorSTUCK = numpy.array([Xsmooth[i,0] - (command[i]+Thresh)])
        errorSTUCK = errorSTUCK/numpy.diag(StdDev)
        probFault[i,1] = 100.0*(1-norm.cdf(errorSTUCK))
        
        faultStatus[i,1] = 0.98 if Xsmooth[i,0] < (command[i]-Thresh) else 0.52
        
        # Check the real faults
        if numpy.abs(command[i]-opening[i]) >= 0.01:
            realFault[i,1] = 1.0
        else:
            realFault[i,1] = 0.0
        realFault[i,0] = 0.0
    
    return (probFault, faultStatus)

def plotFaultStatus(time, faultStatus, ax):
    """
    This function plots the estimated faults statuses and the actual
    faults. This plot is helpful to understand to place in a specific
    context the estimations of the algorithm.
    """
    ax.plot(time, faultStatus[:,0],'k--',label='$F_{Leak}$')
    ax.plot(time, faultStatus[:,1],'k',label='$F_{Stuck}$')
    
    t0 = pd.to_datetime(96.4, unit = "s", utc = True)
    t1 = pd.to_datetime(150.0, unit = "s", utc = True)
    t2 = pd.to_datetime(169, unit = "s", utc = True)
    t3 = pd.to_datetime(360.0, unit = "s", utc = True)
    ax.fill_between([t0, t1], [-10, -10], [0.5,0.5], facecolor='red', interpolate=True, alpha=0.2)
    ax.fill_between([t2, t3], [0.5, 0.5], [10,10], facecolor='red', interpolate=True, alpha=0.2)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Time')
    ax.set_ylabel('Faults')
    
    t4 = pd.to_datetime(200, unit = "s", utc = True)
    t5 = pd.to_datetime(110, unit = "s", utc = True)
    plt.text(t4, 0.75, "Stuck")
    plt.text(t5, 0.25, "Leak")
    plt.tick_params(\
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelleft='off',
        labelbottom='off') # labels along the bottom edge are off
    return

if __name__ == '__main__':
    main()