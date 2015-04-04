'''
Created on Nov 7, 2013

@author: marco
'''
import os
import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FmuUtils import Model
from FmuUtils import CsvReader

def condenser_water_temperature(oat, max_T, min_T, T_ref = 273.15 + 24):
    """
    This function is used to define a value for the condenser water temperature given the outdoor air temperature
    and other parameters like the constraints and a reference value
    """
    N = len(oat)
    for i in range(N):
        oat[i] = np.max([ np.min([ T_ref + 0.6*(oat[i]-(273.15+20)) , max_T]), min_T])
    return oat

def flow_generator(flow_n, time):
    """
    This function computes a variable flow rate using a random process and a moving
    average filter.
    """
    flow = 0.85*flow_n*np.ones(np.shape(time))
    np.random.seed(12)
    noise = 0.15*flow_n*(np.random.rand(len(time)) - 0.5)
    N = 100
    for i in np.arange(N, len(time)):
        flow[i] += np.mean(noise[i-N:i])
    return flow

def main(days = 1):
    
    # Assign an existing FMU to the model, depending on the platform identified
    dir_path = os.path.dirname(__file__)
    if platform.architecture()[0]=="32bit":
        print "32-bit architecture"
        filePath = os.path.join(dir_path, "..", "..","..", "modelica", "FmuExamples", "Resources", "FMUs", "Chiller_dymola2015_etaPL.fmu")
    else:
        print "64-bit architecture"
        filePath = os.path.join(dir_path, "..", "..","..", "modelica", "FmuExamples", "Resources", "FMUs", "ChillerSim_64bit.fmu")
    
    # Initialize the FMU model empty
    m = Model.Model()
    
    # ReInit the model with the new FMU
    m.ReInit(filePath)
    
    # Show details
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
    # Set the CSV file associated to the input
    inputPath = os.path.join(dir_path, "..", "..","..", "modelica", "FmuExamples", "Resources", "data", "Jun11.csv")
    input = m.GetInputByName("On")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("ON")
    
    input = m.GetInputByName("m_flow_CW")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("CD_flow_kgs")
    
    input = m.GetInputByName("m_flow_CH")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("CH_flow_kgs")
    
    input = m.GetInputByName("T_CW_in")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("CD_T_k")
    
    input = m.GetInputByName("T_CH_in")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("CH_T_k")
    
    input = m.GetInputByName("TCHWSet")
    input.GetCsvReader().OpenCSV(inputPath)
    input.GetCsvReader().SetSelectedColumn("CH_SP_k")
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # Reinitialize state variables (temperatures in the volumes, of the sensors, and control action)
    m.SetReal(m.GetVariableObject("chi.vol1.dynBal.medium.T"), 300)
    #m.SetReal(m.GetVariableObject("TCDWlea.T"), 300)
    m.SetReal(m.GetVariableObject("chi.vol2.dynBal.medium.T"), 279)
    #m.SetReal(m.GetVariableObject("TCHWleachi.T"), 279)
    m.SetReal(m.GetVariableObject("conPI.I.y"), 0.83)
    
    # Change the nominal power of the compressor
    m.SetReal(m.GetVariableObject("P_nominal"), 1500e3)
    
    # Decide to use a fixed efficiency or not
    #m.SetReal(m.GetVariableObject("chi.external_etaPL"), False)
    #m.SetReal(m.GetVariableObject("eta_PL"), 0.7)
    
    # Show the values of the state variables
    print "The state vector is:",m.GetState()
    
    ####################################################################################
    # Take the data series from the CSV file and create modified input time series
    JuneData = np.genfromtxt(inputPath, delimiter=",", skip_header=1)
    
    time = pd.to_datetime(JuneData[:,0], unit="s")
    OAT = JuneData[:,14]
    
    ON = JuneData[:,13]
    CD_flow = flow_generator(236.0, time)
    CH_flow = flow_generator(195.0, time)
    CD_T = condenser_water_temperature(OAT, 273.15+28, 273.15+15, 273.15 + 20)
    CH_T = JuneData[:,15]
    SP_T = JuneData[:,17]
    input_data = np.vstack((ON, CD_flow, CD_T, CH_T, CH_flow, SP_T))
    input_data = np.transpose(input_data)
    
    ####################################################################################
    # Simulate
    t0 = pd.to_datetime(0.0, unit = "s")
    t1 = pd.to_datetime(3600.0*24*days, unit = "s")
    time, results = m.Simulate(start_time = t0, final_time = t1, time = time, Input = input_data, complete_res = True)
    
    return (time, results)
    
def showResults(time, results):
    """
    This function shows the results of the simulation
    """
    
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(211)
    ax1.plot(time,results["P"],'g',label='P',alpha=1.0)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('power')
    ax1.set_xlim([time[0], time[-1]])
    legend = ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    
    ax2  = fig1.add_subplot(212)
    ax2.plot(time,results["T_CH_Lea"],'b',label='$T_{CH}^{LEA}$',alpha=1.0)
    ax2.plot(time,results["T_CW_lea"],'r',label='$T_{CW}^{LEA}$',alpha=1.0)
    ax2.plot(time,results["TCHWSet"],'y',label='$T_{SP}$',alpha=1.0)
    ax2.plot(time,results["T_CH_in"],'k',label='$T_{CH}^{IN}$',alpha=1.0)
    ax2.plot(time,results["T_CW_in"],'g',label='$T_{CW}^{IN}$',alpha=1.0)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Output variable')
    ax2.set_xlim([time[0], time[-1]])
    legend = ax2.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax2.grid(False)
    plt.savefig('FirstOrder.pdf',dpi=300, bbox_inches='tight', transparent=True,pad_inches=0.1)
    
    fig2 = plt.figure()
    ax3  = fig2.add_subplot(311)
    ax3.plot(time,results["m_flow_CW"],'k',label='$F_{CW}$',alpha=1.0)
    ax3.plot(time,results["m_flow_CH"],'r',label='$F_{CH}$',alpha=1.0)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Flow')
    ax3.set_xlim([time[0], time[-1]])
    legend = ax3.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax3.grid(False)
    
    ax4  = fig2.add_subplot(312)
    ax4.plot(time,results["chi.ext_COP"],'k',label='$COP$',alpha=1.0)
    ax4.plot(time,results["chi.COPCar"],'r',label='$COP_{CAR}$',alpha=1.0)
    ax4.plot([time[0], time[-1]], [results["chi.COP_nominal"], results["chi.COP_nominal"]],'k--',label='$COP_{n}$',alpha=1.0)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('COP')
    ax4.set_xlim([time[0], time[-1]])
    legend = ax4.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax4.grid(False)
    
    ax5  = fig2.add_subplot(313)
    ax5.plot(time,results["chi.etaPL"],'r',label='$\eta_{PL}$',alpha=1.0)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('$\eta$')
    ax5.set_xlim([time[0], time[-1]])
    legend = ax5.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax5.grid(False)
    
    fig3 = plt.figure()
    ax6  = fig3.add_subplot(111)
    ax6.plot(results["Qcool"]/1000, 3.5168/results["chi.COP"],'ko',label='$COP$',alpha=0.2)
    ax6.plot([0, np.max(results["Qcool"])/1000], [3.5168/results["chi.COP_nominal"], 3.5168/results["chi.COP_nominal"]],'k--',label='$COP_{n}$',alpha=1.0)
    ax6.set_xlabel('Load [kW]')
    ax6.set_ylabel('COP [kW/Ton]')
    ax6.set_ylim([0, 5])
    legend = ax6.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax6.grid(False)
    
    plt.show()

def saveResults(time, results, fileName, samplingTime = 60*5, addNoise = False, noises = {}):
    """
    This function saves the results of the simulation in a csv file specified by the parameter fileName.
    The results will then be used by the state estimation/FDD.
    It can be used also to generate data with noise. The noise levels are specified by a dictionary that has
    the variable name as key and the the noise amplitude as value.
    """
    
    # results to save
    to_save = ["On", "T_CH_Lea", "T_CW_lea", "TCHWSet", "T_CH_in", "T_CW_in", "m_flow_CW",\
               "m_flow_CH", "chi.COP", "chi.etaPL", "P"]
    
    # Create the header of the CSV file
    header = "time, "
    i = 0
    for name in to_save:
        header += name
        if i < len(to_save)-1:
            header +=", "
        else:
            header +="\n"
        i += 1
    
    # Create the sampled time interval
    sampled_time = np.arange(time[0], time[-1]+samplingTime, samplingTime)
    
    # Create matrix with the data, starting with the time column
    M = sampled_time
    for name in to_save:
        sampled_values = np.interp(sampled_time, time, results[name])
        
        # Check if noise should be added
        if addNoise:
            if noises.has_key(name):
                # compute uniform noise and translate it from [0,1] -> [-0.5, 0.5]
                noise = noises[name]*(np.random.rand(len(sampled_time)) - 0.5)
                sampled_values += noise
            else:
                print "Not possible to add noise, the variable:",name,"is not present."
        
        M = np.vstack((M, sampled_values))
        
    # Values are stacked in rows, transpose them
    M = np.transpose(M)
    
    # save data to file
    np.savetxt(fileName, M, fmt = "%.4f", delimiter=",", newline="\n")
    
    # Add the header
    f = open(fileName,"a+")
    lines = f.readlines()
    f.close()
    f = open(fileName, "w")
    lines = [header]+lines
    f.writelines(lines)
    f.close()
        
if __name__ == '__main__':
    
    ####################################################################################
    # Get results from the simulation
    time, results = main(days = 7)
    
    ####################################################################################
    # Show the results
    showResults(time, results)
    
    ####################################################################################
    # Save them in a CSV file for being used during the FDD process
    noises_low = {"T_CH_Lea": 1.0, "T_CW_lea": 1.0, "T_CH_in": 1.0, "T_CW_in":1.0, "m_flow_CW": 5, "m_flow_CH": 5, "P":10000}
    noises_high = {"T_CH_Lea": 2.0, "T_CW_lea": 2.0, "T_CH_in": 2.0, "T_CW_in":2.0, "m_flow_CW": 15, "m_flow_CH": 15, "P":50000}
    saveResults(time, results, "ChillerResults7_noisyHigh.csv", addNoise = True, noises = noises_high)