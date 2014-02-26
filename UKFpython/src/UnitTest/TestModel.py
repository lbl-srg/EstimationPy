'''
Created on Feb 25, 2014

@author: marco
'''
import unittest
from FmuUtils import Model

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_InstantiateModelEmpty(self):
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
    
    def __InstantiateModel(self, reinit = False):
        # Assign an existing FMU to the model
        filePath = "../../modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu"
        
        # Initialize the FMU model
        if reinit:
            m = Model.Model()
            m.ReInit(filePath)
        else:
            m = Model.Model(filePath)
            
        # test default values
        name = "FmuExamples.FirstOrder"
        self.assertEqual(name, m.GetFmuName(), "The FMU name is not: %s" % name)
        
        # Check FMU details
        self.assertIsNotNone(m.GetFMU(), "The FMU object has not to be None")
        self.assertEqual(filePath, m.GetFmuFilePath(), "The FMU file path is not the one specified")
        
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
        
        # Delete the FMU that may cause problems when reloading it for other tests
        m.unloadFMU()
    
    def test_InstantiateModel(self):
        self.__InstantiateModel(reinit = False)
    
    def test_InstantiateModelReinit(self):
        self.__InstantiateModel(reinit = False)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()