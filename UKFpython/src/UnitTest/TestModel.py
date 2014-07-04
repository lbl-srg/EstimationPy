'''
Created on Feb 25, 2014

@author: marco
'''
import unittest
import platform
import os
import pandas as pd
import numpy as np

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
        The initialization can be don ewhen creating the instance or calling the ReInit method.
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
        ind = pd.date_range('2000-1-1', periods = 30, freq='s')
        ds = pd.Series(np.ones(30), index = ind)
        
        # Set the CSV file associated to the input
        inp = m.GetInputByName("u")
        inp.SetDataSeries(ds)
    
        # Initialize the model for the simulation
        m.InitializeSimulator()
        
        # Simulate
        time, results = m.Simulate()
        
        
    
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()