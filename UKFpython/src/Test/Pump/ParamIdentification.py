'''
Created on Nov 7, 2013

@author: marco
'''
import numpy
import matplotlib.pyplot as plt

from FmuUtils import Model
from ukf.ukfFMU import ukfFMU



def main():
    
    # Assign an existing FMU to the model
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/Pump_MBL3.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-4, rtol=1e-3)
    
    # Path of the csv file containing the data series
    #csvPath = "../../../modelica/FmuExamples/Resources/data/DataPumpVeryShort.csv"
    #csvPath = "../../../modelica/FmuExamples/Resources/data/DataPump_16to19_Oct2012.csv"
    csvPath = "../../../modelica/FmuExamples/Resources/data/DataPump_16to19_Oct2012_variableStep.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("Nrpm")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("Pump.Speed")
    
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("P_el")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("Pump.kW")
    output.SetMeasuredOutput()
    output.SetCovariance(0.15)
    
    """
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("V_flow")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("Pump.gpm")
    output.SetMeasuredOutput()
    output.SetCovariance(50.0)
    """
    p_start = [0.30000000, 0.40000000, 0.60000000, 0.8] # 0
    #p_start = [0.20000000, 0.40000000, 0.60000000, 0.8] # 0
    #p_start = [0.28538255, 0.50017618, 0.71668581, 1.] # 7
    #p_start = [0.21517598, 0.50276898, 0.79260251, 1.] # 25
    #p_start = [0.19080228, 0.50549745, 0.81889495, 1.] # 35
    #p_start = [0.11824324, 0.5869894,  0.8849001,  1.]
    #p_start = [0.11574544, 0.67699191, 0.8862938, 1.]
    
    cov = 0.1
    
    #################################################################
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("pump.power.P[1]"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[0]
    par.SetInitialValue(p_start[0])
    par.SetCovariance(cov)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("pump.power.P[2]"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[1]
    par.SetInitialValue(p_start[1])
    par.SetCovariance(cov)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("pump.power.P[3]"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[2]
    par.SetInitialValue(p_start[2])
    par.SetCovariance(cov)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("pump.power.P[4]"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[3]
    par.SetInitialValue(p_start[3])
    par.SetCovariance(cov)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(1.0)
    par.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    ukf_FMU.setUKFparams()
    
    pars = ukf_FMU.ParameterEstimation(maxIter = 500)
    print pars
    
    ShowResults(pars)
      
def ShowResults(pars):
    
    pars = numpy.squeeze(numpy.array(pars))
    
    fig0 = plt.figure()
    fig0.set_size_inches(12,8)
    ax0  = fig0.add_subplot(111)
    ax0.plot(pars, label='$pars$',alpha=1.0)
    plt.show()
    
if __name__ == '__main__':
    main()