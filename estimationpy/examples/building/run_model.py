"""
Created on Nov 7, 2013

@author: marco
"""
import os
import platform
import pytz

import matplotlib.pyplot as plt
import pandas as pd

from collections import OrderedDict
from estimationpy.fmu_utils.model import Model

def main():
    
    # Initialize the FMU model empty
    m = Model()
    
    # Assign an existing FMU to the model, depending on the platform identified
    dir_path = os.path.dirname(__file__)
    
    # Define the path of the FMU file
    if platform.architecture()[0]=="32bit":
        print "32-bit architecture"
        filePath = os.path.join(dir_path, "model", "model_se.fmu")
    else:
        print "64-bit architecture"
        filePath = os.path.join(dir_path, "model", "model_se.fmu")
    
    # ReInit the model with the new FMU
    m.re_init(filePath)
    
    # Show details
    print m
    
    # Show the inputs
    print "The names of the FMU inputs are: ", m.get_input_names(), "\n"
    
    # Show the outputs
    print "The names of the FMU outputs are:", m.get_output_names(), "\n"

    # Set the parameters of the model
    fixpars = OrderedDict([("bui.hva.cHea", 812249.38954445894), ("bui.hva.cTecRoo", 31682287.202939499),
                           ("bui.hva.capHea.TSta", 292.33922569546098), ("bui.hva.capTecRoo.TSta", 293.48308013220799),
                           ("bui.hva.hp1.Intercept", 2.61214752617), ("bui.hva.hp1.TAmb7", 0.054522645537),
                           ("bui.hva.hp1.TSup35", -0.0123192467622), ("bui.hva.hp1.E6500", -0.0001176597066),
                           ("bui.hva.hp1.TAmb7_E6500", -1.53890877556e-05), ("bui.hva.hp2.Intercept", 2.58406557762),
                           ("bui.hva.hp2.TAmb7", 0.0384068602888), ("bui.hva.hp2.TSup35", -0.025053392321),
                           ("bui.hva.hp2.E6500", -0.000141527731896), ("bui.hva.hp2.TAmb7_E6500", -1.50277640388e-05),
                           ("bui.hva.rLos", 0.0087779013329769198),
                           ("bui.hva.rTecRoo", 0.0050198340105892499), ("bui.hva.gb.Intercept", 0.872048186735),
                           ("bui.hva.gb.Q86000", -8.84828553083e-07), ("bui.hva.gb.TAmb7", 0.00481677713153)])
    fixpars.update(OrderedDict([("bui.use.nOcc", 136.67024436265001), ("bui.ven.mFloVen", 1.1072098104391299),
                                ("bui.zon.cInt", 455341540.68018103), ("bui.zon.cZon", 111540557.533288),
                                ("bui.zon.capInt.TSta", 293.61372904158401), ("bui.zon.capZon.TSta", 291.892741044606),
                                ("bui.zon.rInt", 0.00019259685456255999), ("bui.zon.rWal", 0.00094286602169893301),
                                ("bui.hva.fraRad", 0.0)]))

    # Set the values for the parameters
    m.get_fmu().reset()
    m.get_fmu().set(fixpars.keys(),fixpars.values())

    # Set the CSV file associated to the input
    path_monitoring_data = os.path.join(dir_path, "data", "dataset_simulation.csv")

    BXL = pytz.timezone("Europe/Brussels")
    df_data = pd.read_csv(path_monitoring_data, index_col=0, header=0, parse_dates=True)
    df_data = df_data.resample('900S')
    
    # Link the columns of the CSV file to the inputs
    for n in ["Q_HP1", "Q_HP2", "Q_GB", "TAmb", "I_GloHor_sat", "powEle", "QCon", "prfOcc", "prfVen"]:
        input = m.get_input_by_name(n)
        input.set_data_series(df_data[n])

    # Initialize the model for the simulation
    m.initialize_simulator()
                      
    # Simulate
    time, results = m.simulate()
    
    # Show the results and forward the data frame containing the real data
    show_results(time, results, df_data)

    return time, results
    
def show_results(time, results, df_data):
    
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(111)
    ax1.plot(time,results["TZon"],"g",label="$T_{Zone}^S$",alpha=1.0)
    ax1.plot(df_data.index,df_data["TZon"],"g--",label="$T_{Zone}$",alpha=1.0)
    ax1.plot(time,results["T_emi_sup"],"r",label="$T_{EMI}^S$",alpha=1.0)
    ax1.plot(df_data.index,df_data["T_EMI_sup"],"r--",label="$T_{EMI}$",alpha=1.0)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature")
    ax1.set_xlim([time[0], time[-1]])
    legend = ax1.legend(loc="upper center",bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True, shadow=True)
    legend.draggable()
    ax1.grid(False)
    plt.show()
   
if __name__ == "__main__":
    main()