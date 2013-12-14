'''
Created on Nov 7, 2013

@author: marco
'''
import numpy
from FmuUtils import Model
from ukf.ukfFMU import ukfFMU

import matplotlib.pyplot as plt

def main():
    
    # Assign an existing FMU to the model
    filePath = "../../../modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath, atol=1e-6, rtol=1e-6)
    
    # Path of the csv file containing the data series
    csvPath = "../../../modelica/FmuExamples/Resources/data/NoisySimulationData_FirstOrder.csv"
    
    # Set the CSV file associated to the input, and its covariance
    input = m.GetInputByName("u")
    input.GetCsvReader().OpenCSV(csvPath)
    input.GetCsvReader().SetSelectedColumn("system.u")
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("y")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("system.y")
    output.SetMeasuredOutput()
    output.SetCovariance(0.5)
    
    # Set the CSV file associated to the output, and its covariance
    output = m.GetOutputByName("x")
    output.GetCsvReader().OpenCSV(csvPath)
    output.GetCsvReader().SetSelectedColumn("system.x")
    output.SetMeasuredOutput()
    output.SetCovariance(0.5)
    
    #################################################################
    # Select the state to be identified
    m.AddVariable(m.GetVariableObject("x"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetVariables()[0]
    par.SetInitialValue(1.8)
    par.SetCovariance(0.5)
    
    #################################################################
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("a"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[0]
    par.SetInitialValue(-0.5)
    par.SetCovariance(0.01)
    par.SetMinValue(-10.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(-0.1)
    par.SetConstraintHigh(True)
    
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("b"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[1]
    par.SetInitialValue(0.5)
    par.SetCovariance(0.01)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(20.0)
    par.SetConstraintHigh(True)
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("c"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[2]
    par.SetInitialValue(0.5)
    par.SetCovariance(0.01)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(20.0)
    par.SetConstraintHigh(True)
    
    # Select the parameter to be identified
    m.AddParameter(m.GetVariableObject("d"))
    
    # Set initial value of parameter, and its covariance and the limits (if any)
    par = m.GetParameters()[3]
    par.SetInitialValue(0.5)
    par.SetCovariance(0.01)
    par.SetMinValue(0.0)
    par.SetConstraintLow(True)
    par.SetMaxValue(20.0)
    par.SetConstraintHigh(True)
    
    # Initialize the model for the simulation
    m.InitializeSimulator()
    
    # instantiate the UKF for the FMU
    ukf_FMU = ukfFMU(m, augmented = False)
    ukf_FMU.setUKFparams(0.05, 2, 1)
    
    pars = ukf_FMU.ParameterEstimation(maxIter = 600)
    print pars
    
    ShowResults(pars)
      
def ShowResults(pars):
    
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