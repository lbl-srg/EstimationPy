'''
Created on July 3, 2014

@author: marco
'''
import unittest
import platform
import os
import pytz
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from estimationpy.fmu_utils import model

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
            self.filePath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder.fmu")
        else:
            print "64-bit architecture"
            self.filePath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_64bit.fmu")
            
        # Path of the CSV data
        self.csv_inputPath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")

    def test_instantiate_model_empty(self):
        """
        This function tests the initialization of a model that has not an FMU associated to it
        """
        # Instantiate an empty model
        m = model.Model()
        
        # test default values
        self.assertEqual("", m.get_fmu_name(), "The FMU name has to be empty")
        
        # Check FMU details
        self.assertIsNone(m.get_fmu(), "The FMU object has to be None")
        self.assertIsNone(m.get_fmu_file_path(), "The FMU file path has to be None")
        # The properties are
        # (self.name, self.author, self.description, self.type, self.version, self.guid, self.tool, self.numStates)
        self.assertEqual(("", "", "", "", "", "", "", ""), m.get_properties(), "The property values have to be all empty")
        
        # Check list initialized correctly
        self.assertListEqual([], m.get_inputs(), "The list of inputs has to be empty")
        self.assertListEqual([], m.get_outputs(), "The list of outputs has to be empty")
        self.assertListEqual([], m.get_input_names(), "The list of input names has to be empty")
        self.assertListEqual([], m.get_output_names(), "The list of output names has to be empty")
        self.assertListEqual([], m.get_parameters(), "The list of parameters has to be empty")
        self.assertListEqual([], m.get_parameter_names(), "The list of parameters names has to be empty")
        self.assertListEqual([], m.get_variables(), "The list of variables has to be empty")
        self.assertListEqual([], m.get_variable_names(), "The list of variables names has to be empty")
        
        # Check functions counting the list items work correctly
        self.assertEqual(0, m.get_num_inputs(), "The number of inputs has to be zero")
        self.assertEqual(0, m.get_num_outputs(), "The number of outputs has to be zero")
        self.assertEqual(0, m.get_num_measured_outputs(), "The number of measured outputs has to be zero")
        self.assertEqual(0, m.get_num_parameters(), "The number of parameters has to be zero")
        self.assertEqual(0, m.get_num_variables(), "The number of variables has to be zero")
        self.assertEqual(0, m.get_num_states(), "The number of state variables has to be zero")
        
        # test access to FMI methods
        self.assertIsNone(m.get_variable_object("a"), "trying to access a variable object should return None") 
    
    def __instantiate_model(self, reinit = False):
        """
        This function tests the initialization of a model given an FMU.
        The initialization can be done when creating the instance or after by calling the ReInit method.
        """
        
        # Initialize the FMU model
        if reinit:
            m = model.Model()
            m.re_init(self.filePath)
        else:
            m = model.Model(self.filePath)
            
        # test default values
        name = "FmuExamples.FirstOrder"
        self.assertEqual(name, m.get_fmu_name(), "The FMU name is not: %s" % name)
        
        # Check FMU details
        self.assertIsNotNone(m.get_fmu(), "The FMU object has not to be None")
        self.assertEqual(self.filePath, m.get_fmu_file_path(), "The FMU file path is not the one specified")
        
        # Check list initialized correctly
        self.assertListEqual(['u'], m.get_input_names(), "The list of input names is not correct ")
        self.assertListEqual(['y','x'], m.get_output_names(), "The list of output names is not correct")
        
        # Check functions counting the list items work correctly
        self.assertEqual(1, m.get_num_inputs(), "The number of inputs has to be one")
        self.assertEqual(2, m.get_num_outputs(), "The number of outputs has to be two")
        self.assertEqual(0, m.get_num_measured_outputs(), "The number of measured outputs has to be zero")
        self.assertEqual(0, m.get_num_parameters(), "The number of parameters has to be zero")
        self.assertEqual(0, m.get_num_variables(), "The number of variables has to be zero")
        self.assertEqual(1, m.get_num_states(), "The number of state variables has to be zero")
        
        # Check getting inputs and output objects
        self.assertIsNotNone(m.get_input_by_name("u"), "The object corresponding to input 'u' should be accessible")
        self.assertIsNone(m.get_input_by_name("y"), "The object corresponding to input 'y' should not be accessible (its an output)")
        self.assertIsNotNone(m.get_output_by_name("y"), "The object corresponding to output 'y' should be accessible")
        self.assertIsNotNone(m.get_output_by_name("x"), "The object corresponding to output 'x' should be accessible")
        self.assertIsNone(m.get_output_by_name("u"), "The object corresponding to output 'u' should not be accessible (its an input)")
    
    def test_instantiate_model(self):
        """
        Model that tests the initialization of a model given an FMU during instantiation
        """
        self.__instantiate_model(reinit = False)
    
    def test_instantiate_model_re_init(self):
        """
        Model that tests the initialization of a model given an FMU after the instantiation
        """
        self.__instantiate_model(reinit = True)
    
    def test_initialize_model(self):
        """
        This test is check the initialization of a model
        """
        # Initialize the FMU model empty
        m = model.Model()
    
        # ReInit the model with the new FMU
        m.re_init(self.filePath)
    
        # Show details
        print m
        
        # Show the inputs
        print "The names of the FMU inputs are: ", m.get_input_names(), "\n"
        
        # Show the outputs
        print "The names of the FMU outputs are:", m.get_output_names(), "\n"
    
        # Set the CSV file associated to the input
        inp = m.get_input_by_name("u")
        inp.get_csv_reader().open_csv(self.csv_inputPath)
        inp.get_csv_reader().set_selected_column("system.u")
    
        # Initialize the model for the simulation
        m.initialize_simulator()
        
    def test_run_model_CSV(self):
        """
        This function tests if the model can be run when loading data from a csv file
        """
        # Initialize the FMU model empty
        m = model.Model()
    
        # ReInit the model with the new FMU
        m.re_init(self.filePath)
    
        # Set the CSV file associated to the input
        inp = m.get_input_by_name("u")
        inp.get_csv_reader().open_csv(self.csv_inputPath)
        inp.get_csv_reader().set_selected_column("system.u")
    
        # Initialize the model for the simulation
        m.initialize_simulator()
        
        # Simulate
        time, results = m.simulate()
        # Compare the results with the expected ones. Given the default 
        # values of the parameters a = -1, b = 2.5, c = 3.0, d = 0.1,
        
        # Read the simulation time vector
        self.assertEqual(pd.to_datetime(0.0, unit = "s", utc = True), time[0], "The initial time does not correspond")
        self.assertEqual(pd.to_datetime(30.0, unit = "s", utc = True), time[-1], "The final time does not correspond")
        
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
        m = model.Model()
    
        # ReInit the model with the new FMU
        m.re_init(self.filePath)
        
        # Create a pandas.Series for the input u
        ind = pd.date_range('2000-1-1', periods = 31, freq='s', tz = pytz.utc)
        ds = pd.Series(np.ones(31), index = ind)
        
        # Set the CSV file associated to the input
        inp = m.get_input_by_name("u")
        inp.set_data_series(ds)
    
        # Set parameters a, b, c, d of the model
        par_a = m.get_variable_object("a")
        m.set_real(par_a, -1.0)
        par_b = m.get_variable_object("b")
        m.set_real(par_b, 4.0)
        par_c = m.get_variable_object("c")
        m.set_real(par_c, 6.0)
        par_d = m.get_variable_object("d")
        m.set_real(par_d, 0.0)
        
        # Initialize the model for the simulation
        m.initialize_simulator()
        
        # Read the values that have just been set
        self.assertEqual(-1.0, m.get_real(par_a), "Parameter a of the FMU has to be equal to -1.0")
        self.assertEqual(4.0, m.get_real(par_b), "Parameter b of the FMU has to be equal to 4.0")
        self.assertEqual(6.0, m.get_real(par_c), "Parameter c of the FMU has to be equal to 6.0")
        self.assertEqual(0.0, m.get_real(par_d), "Parameter d of the FMU has to be equal to 0.0")
        
        # Simulate using start and final time of type datetime.datetime
        t0 = datetime(2000, 1, 1, 0, 0, 10, tzinfo = pytz.utc)
        t1 = datetime(2000, 1, 1, 0, 0, 25, tzinfo = pytz.utc)
        time, results = m.simulate(start_time = t0, final_time = t1)
        
        # Read the simulation time vector
        self.assertTrue(t0 - time[0] < timedelta(0, 1e-6) or
                        time[0] - t0 < timedelta(0, 1e-6),
                        "The initial time does not correspond: {0} != {1}".format(t0, time[0]))
        self.assertTrue(t1 - time[-1] < timedelta(0, 1e-6) or
                        time[-1] - t1 < timedelta(0, 1e-6),
                        "The final time does not correspond: {0} != {1}".format(t1, time[-1]))
        
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
        m = model.Model()
    
        # ReInit the model with the new FMU
        m.re_init(self.filePath)
        
        # Create a pandas.Series for the input u
        ind = pd.date_range('2000-1-1', periods = 31, freq='s', tz = pytz.utc)
        ds = pd.Series(np.ones(31), index = ind)
        
        # Set the CSV file associated to the input
        inp = m.get_input_by_name("u")
        inp.set_data_series(ds)
        
        # Initialize the model for the simulation with start time that are not correct
        self.assertRaises(TypeError, m.initialize_simulator, 0.0)
        self.assertRaises(TypeError, m.initialize_simulator, [0.0])
        self.assertRaises(TypeError, m.initialize_simulator, "2000-1-1 00:00:00")
        self.assertRaises(IndexError, m.initialize_simulator, datetime(1999, 12, 31, 23, 59, 59, tzinfo = pytz.utc))
        self.assertRaises(IndexError, m.initialize_simulator, datetime(2000, 1, 1, 0, 0, 30, 1000, tzinfo = pytz.utc))
        
        # Initialize with a correct value
        self.assertTrue(m.initialize_simulator(datetime(2000, 1, 1, 0, 0, 0, tzinfo = pytz.utc)), \
                        "The model has not been correctly initialized")
        
    def test_model_simulate_exceptions(self):
        """
        This function tests if the exceptions generated by the simulate method are correctly 
        generated when needed.
        """
        
        # Initialize the FMU model empty
        m = model.Model()
    
        # ReInit the model with the new FMU
        m.re_init(self.filePath)
        
        # Create a pandas.Series for the input u
        ind = pd.date_range('2000-1-1', periods = 31, freq='s', tz=pytz.utc)
        ds = pd.Series(np.ones(31), index = ind)
        
        # Set the CSV file associated to the input
        inp = m.get_input_by_name("u")
        inp.set_data_series(ds)
        
        # Initialize with a correct value
        m.initialize_simulator(datetime(2000, 1, 1, 0, 0, 0, tzinfo = pytz.utc))
        
        # Try to simulate passing wrong values
        self.assertRaises(TypeError, m.simulate, 0)
        self.assertRaises(TypeError, m.simulate, [0.0])
        self.assertRaises(TypeError, m.simulate, "0.0")
        self.assertRaises(IndexError, m.simulate, datetime(1999, 12, 31, 23, 59, 59, tzinfo = pytz.utc))
        self.assertRaises(IndexError, m.simulate, datetime(2000, 1, 1, 0, 0, 30, 1000, tzinfo = pytz.utc))
        
        self.assertRaises(TypeError, m.simulate, None, 0)
        self.assertRaises(TypeError, m.simulate, None, [0.0])
        self.assertRaises(TypeError, m.simulate, None, "0.0")
        self.assertRaises(IndexError, m.simulate, None, datetime(1999, 12, 31, 23, 59, 59, tzinfo = pytz.utc))
        self.assertRaises(IndexError, m.simulate, None, datetime(2000, 1, 1, 0, 0, 30, 1000, tzinfo = pytz.utc))
        
        self.assertRaises(IndexError, m.simulate, \
                          datetime(2000, 1, 1, 0, 0, 10, tzinfo = pytz.utc), \
                          datetime(2000, 1, 1, 0, 0, 5, tzinfo = pytz.utc))
    
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
