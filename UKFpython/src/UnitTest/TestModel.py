'''
Created on July 3, 2014

@author: marco
'''
import unittest
import platform
import os
import pandas as pd
import numpy as np

from datetime import datetime
from FmuUtils import Model

class Test(unittest.TestCase):

    def setUp(self):
        """
        Initialize the class for testing the Model
        """
        # Assign an existing FMU to the model, depending on the platform identified
        dir_path = os.path.dirname(__file__)
        
        # Define the path of the FMU file
        if platform.architecture()[0]=="32bit":
            print "32-bit architecture"
            self.filePath = os.path.join(dir_path, "..","..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder.fmu")
        else:
            print "64-bit architecture"
            self.filePath = os.path.join(dir_path, "..","..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_64bit.fmu")
            
        # Path of the CSV data
        self.csv_inputPath = os.path.join(dir_path, "..","..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")

    def test_instantiate_model_empty(self):
        """
        This function tests the initialization of a model that has not an FMU associated to it
        """
        # Instantiate an empty model
        m = Model.Model()
        
        # test default values
        self.assertEqual("", m.GetFmuName(), "The FMU name has to be empty")
        
        # Check FMU details
        self.assertIsNone(m.GetFMU(), "The FMU object has to be None")
        self.assertIsNone(m.GetFmuFilePath(), "The FMU file path has to be None")
        # The properties are
        # (self.name, self.author, self.description, self.type, self.version, self.guid, self.tool, self.numStates)
        self.assertEqual(("", "", "", "", "", "", "", ""), m.GetProperties(), "The property values have to be all empty")
        
        # Check list initialized correctly
        self.assertListEqual([], m.GetInputs(), "The list of inputs has to be empty")
        self.assertListEqual([], m.GetOutputs(), "The list of outputs has to be empty")
        self.assertListEqual([], m.GetInputNames(), "The list of input names has to be empty")
        self.assertListEqual([], m.GetOutputNames(), "The list of output names has to be empty")
        self.assertListEqual([], m.GetParameters(), "The list of parameters has to be empty")
        self.assertListEqual([], m.GetParameterNames(), "The list of parameters names has to be empty")
        self.assertListEqual([], m.GetVariables(), "The list of variables has to be empty")
        self.assertListEqual([], m.GetVariableNames(), "The list of variables names has to be empty")
        
        # Check functions counting the list items work correctly
        self.assertEqual(0, m.GetNumInputs(), "The number of inputs has to be zero")
        self.assertEqual(0, m.GetNumOutputs(), "The number of outputs has to be zero")
        self.assertEqual(0, m.GetNumMeasuredOutputs(), "The number of measured outputs has to be zero")
        self.assertEqual(0, m.GetNumParameters(), "The number of parameters has to be zero")
        self.assertEqual(0, m.GetNumVariables(), "The number of variables has to be zero")
        self.assertEqual(0, m.GetNumStates(), "The number of state variables has to be zero")
        
        # test access to FMI methods
        self.assertIsNone(m.GetVariableObject("a"), "trying to access a variable object should return None") 
    
    def __instantiate_model(self, reinit = False):
        """
        This function tests the initialization of a model given an FMU.
        The initialization can be done when creating the instance or after by calling the ReInit method.
        """
        
        # Initialize the FMU model
        if reinit:
            m = Model.Model()
            m.ReInit(self.filePath)
        else:
            m = Model.Model(self.filePath)
            
        # test default values
        name = "FmuExamples.FirstOrder"
        self.assertEqual(name, m.GetFmuName(), "The FMU name is not: %s" % name)
        
        # Check FMU details
        self.assertIsNotNone(m.GetFMU(), "The FMU object has not to be None")
        self.assertEqual(self.filePath, m.GetFmuFilePath(), "The FMU file path is not the one specified")
        
        # Check list initialized correctly
        self.assertListEqual(['u'], m.GetInputNames(), "The list of input names is not correct ")
        self.assertListEqual(['y','x'], m.GetOutputNames(), "The list of output names is not correct")
        
        # Check functions counting the list items work correctly
        self.assertEqual(1, m.GetNumInputs(), "The number of inputs has to be one")
        self.assertEqual(2, m.GetNumOutputs(), "The number of outputs has to be two")
        self.assertEqual(0, m.GetNumMeasuredOutputs(), "The number of measured outputs has to be zero")
        self.assertEqual(0, m.GetNumParameters(), "The number of parameters has to be zero")
        self.assertEqual(0, m.GetNumVariables(), "The number of variables has to be zero")
        self.assertEqual(1, m.GetNumStates(), "The number of state variables has to be zero")
        
        # Check getting inputs and output objects
        self.assertIsNotNone(m.GetInputByName("u"), "The object corresponding to input 'u' should be accessible")
        self.assertIsNone(m.GetInputByName("y"), "The object corresponding to input 'y' should not be accessible (its an output)")
        self.assertIsNotNone(m.GetOutputByName("y"), "The object corresponding to output 'y' should be accessible")
        self.assertIsNotNone(m.GetOutputByName("x"), "The object corresponding to output 'x' should be accessible")
        self.assertIsNone(m.GetOutputByName("u"), "The object corresponding to output 'u' should not be accessible (its an input)")
    
    def test_instantiate_model(self):
        """
        Model that tests the initialization of a model given an FMU during instantiation
        """
        self.__instantiate_model(reinit = False)
    
    def test_instantiate_model_reinit(self):
        """
        Model that tests the initialization of a model given an FMU after the instantiation
        """
        self.__instantiate_model(reinit = True)
    
    def test_initialize_model(self):
        """
        This test is check the initialization of a model
        """
        # Initialize the FMU model empty
        m = Model.Model()
    
        # ReInit the model with the new FMU
        m.ReInit(self.filePath)
    
        # Show details
        print m
        
        # Show the inputs
        print "The names of the FMU inputs are: ", m.GetInputNames(), "\n"
        
        # Show the outputs
        print "The names of the FMU outputs are:", m.GetOutputNames(), "\n"
    
        # Set the CSV file associated to the input
        inp = m.GetInputByName("u")
        inp.GetCsvReader().OpenCSV(self.csv_inputPath)
        inp.GetCsvReader().SetSelectedColumn("system.u")
    
        # Initialize the model for the simulation
        m.InitializeSimulator()
        
    def test_run_model_CSV(self):
        """
        This function tests if the model can be run when loading data from a csv file
        """
        # Initialize the FMU model empty
        m = Model.Model()
    
        # ReInit the model with the new FMU
        m.ReInit(self.filePath)
    
        # Set the CSV file associated to the input
        inp = m.GetInputByName("u")
        inp.GetCsvReader().OpenCSV(self.csv_inputPath)
        inp.GetCsvReader().SetSelectedColumn("system.u")
    
        # Initialize the model for the simulation
        m.InitializeSimulator()
        
        # Simulate
        time, results = m.Simulate()
        # Compare the results with the expected ones. Given the default 
        # values of the parameters a = -1, b = 2.5, c = 3.0, d = 0.1,
        
        # Read the simulation time vector
        self.assertEqual(pd.to_datetime(0.0, unit = "s"), time[0], "The initial time does not correspond")
        self.assertEqual(pd.to_datetime(30.0, unit = "s"), time[-1], "The final time does not correspond")
        
        # Read the results of the simulation
        # x' = -1*x + 2.5*u
        # y  = +3*x + 0.1*u
        # Given the input u = 1 with t in [0, 15) then u = 2 with t in [15,30], at steady state
        # x ~ 5 and y ~ 15.2
        self.assertAlmostEqual(5.0, results["x"][-1], 2, "The steady state value of the \
        state variable x is not 5.0 but %.8f" % (results["x"][-1]))
        self.assertAlmostEqual(15.2, results["y"][-1], 2, "The steady state value of \
        the output variable y is not 15.2 but %.8f" % (results["y"][-1]))
        
    def test_run_model_data_series(self):
        """
        This function tests if the model can be run when loading data form a pandas
        data series
        """
        
        # Initialize the FMU model empty
        m = Model.Model()
    
        # ReInit the model with the new FMU
        m.ReInit(self.filePath)
        
        # Create a pandas.Series for the input u
        ind = pd.date_range('2000-1-1', periods = 31, freq='s')
        ds = pd.Series(np.ones(31), index = ind)
        
        # Set the CSV file associated to the input
        inp = m.GetInputByName("u")
        inp.SetDataSeries(ds)
    
        # Set parameters a, b, c, d of the model
        par_a = m.GetVariableObject("a")
        m.SetReal(par_a, -1.0)
        par_b = m.GetVariableObject("b")
        m.SetReal(par_b, 4.0)
        par_c = m.GetVariableObject("c")
        m.SetReal(par_c, 6.0)
        par_d = m.GetVariableObject("d")
        m.SetReal(par_d, 0.0)
        
        # Initialize the model for the simulation
        m.InitializeSimulator()
        
        # Read the values that have just been set
        self.assertEqual(-1.0, m.GetReal(par_a), "Parameter a of the FMU has to be equal to -1.0")
        self.assertEqual(4.0, m.GetReal(par_b), "Parameter b of the FMU has to be equal to 4.0")
        self.assertEqual(6.0, m.GetReal(par_c), "Parameter c of the FMU has to be equal to 6.0")
        self.assertEqual(0.0, m.GetReal(par_d), "Parameter d of the FMU has to be equal to 0.0")
        
        # Simulate using start and final time of type datetime.datetime
        t0 = datetime(2000, 1, 1, 0, 0, 10)
        t1 = datetime(2000, 1, 1, 0, 0, 25)
        time, results = m.Simulate(start_time = t0, final_time = t1)
        
        # Read the simulation time vector
        self.assertEqual(t0, time[0], "The initial time does not correspond")
        self.assertEqual(t1, time[-1], "The final time does not correspond")
        
        # Read the results of the simulation
        # x' = -1*x + 4*u
        # y  = +6*x + 0*u
        # Given the input u = 1, at steady state
        # x ~ 4 and y ~ 24
        self.assertAlmostEqual(4.0, results["x"][-1], 4, "The steady state value of the \
        state variable x is not 4.0 but %.8f" % (results["x"][-1]))
        self.assertAlmostEqual(24.0, results["y"][-1], 4, "The steady state value of \
        the output variable y is not 24.0 but %.8f" % (results["y"][-1]))
        
    def test_model_init_exceptions(self):
        """
        This function tests if the model can raises exceptions in a proper way when parameters are not
        specified in a proper way
        """
        
        # Initialize the FMU model empty
        m = Model.Model()
    
        # ReInit the model with the new FMU
        m.ReInit(self.filePath)
        
        # Create a pandas.Series for the input u
        ind = pd.date_range('2000-1-1', periods = 31, freq='s')
        ds = pd.Series(np.ones(31), index = ind)
        
        # Set the CSV file associated to the input
        inp = m.GetInputByName("u")
        inp.SetDataSeries(ds)
        
        # Initialize the model for the simulation with start time that are not correct
        self.assertRaises(TypeError, m.InitializeSimulator, 0.0)
        self.assertRaises(TypeError, m.InitializeSimulator, [0.0])
        self.assertRaises(TypeError, m.InitializeSimulator, "2000-1-1 00:00:00")
        self.assertRaises(IndexError, m.InitializeSimulator, datetime(1999, 12, 31, 23, 59, 59))
        self.assertRaises(IndexError, m.InitializeSimulator, datetime(2000, 1, 1, 0, 0, 30, 1000))
        
        # Initialize with a correct value
        self.assertTrue(m.InitializeSimulator(datetime(2000, 1, 1, 0, 0, 0)), "The model has not been correctly initialized")
        
    def test_model_simulate_exceptions(self):
        """
        This function tests if the exceptions generated by the simulate method are correctly 
        generated when needed.
        """
        
        # Initialize the FMU model empty
        m = Model.Model()
    
        # ReInit the model with the new FMU
        m.ReInit(self.filePath)
        
        # Create a pandas.Series for the input u
        ind = pd.date_range('2000-1-1', periods = 31, freq='s')
        ds = pd.Series(np.ones(31), index = ind)
        
        # Set the CSV file associated to the input
        inp = m.GetInputByName("u")
        inp.SetDataSeries(ds)
        
        # Initialize with a correct value
        m.InitializeSimulator(datetime(2000, 1, 1, 0, 0, 0))
        
        # Try to simulate passing wrong values
        self.assertRaises(TypeError, m.Simulate, 0)
        self.assertRaises(TypeError, m.Simulate, [0.0])
        self.assertRaises(TypeError, m.Simulate, "0.0")
        self.assertRaises(IndexError, m.Simulate, datetime(1999, 12, 31, 23, 59, 59))
        self.assertRaises(IndexError, m.Simulate, datetime(2000, 1, 1, 0, 0, 30, 1000))
        
        self.assertRaises(TypeError, m.Simulate, None, 0)
        self.assertRaises(TypeError, m.Simulate, None, [0.0])
        self.assertRaises(TypeError, m.Simulate, None, "0.0")
        self.assertRaises(IndexError, m.Simulate, None, datetime(1999, 12, 31, 23, 59, 59))
        self.assertRaises(IndexError, m.Simulate, None, datetime(2000, 1, 1, 0, 0, 30, 1000))
        
        self.assertRaises(IndexError, m.Simulate, datetime(2000, 1, 1, 0, 0, 10), datetime(2000, 1, 1, 0, 0, 5))
    
    def test_model_not_aligned_inputs(self):
        """
        This function tests the behavior of the model when input data series that are not aligned are 
        provided to it. The inputs data series should be cut in order to select the minimum time
        frame for which all the data are available. Also, the number of point selected within the time frame
        is selected in order to keep the highest sampling frequency
        """
        pass 
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()