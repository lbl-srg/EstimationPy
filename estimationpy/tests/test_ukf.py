'''
Created on Feb 25, 2014

@author: marco
'''
import os
import unittest
import platform
import pytz
from datetime import datetime
import numpy as np
import pandas as pd
from estimationpy.ukf.ukf_fmu import UkfFmu
from estimationpy.fmu_utils.model import Model

dir_path = os.path.dirname(__file__)

class Test(unittest.TestCase):
    """
    This class contains tests to verify the correctness
    of functionalities provided by object of type
    :class:`estimationpy.ukf.ukf_fmu.UkfFmu`.
    """

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def set_first_order_model(self):
        """
        This method is used in different tests and it
        loads the model of the first order system.
        """
        # Define the path of the FMU file
        if platform.architecture()[0]=="32bit":
            filePath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder.fmu")
        else:
            filePath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "FMUs", "FirstOrder_64bit.fmu")
        
        # Initialize the FMU model empty
        self.m = Model(filePath)

    def set_first_order_model_input_outputs(self):
        """
        This methos associates the input and the measured output of
        the first order model from a CSV file.
        """
        # Path of the csv file containing the data series
        csvPath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "data", "NoisySimulationData_FirstOrder.csv")
    
        # Set the CSV file associated to the input, and its covariance
        input_u = self.m.get_input_by_name("u")
        input_u.get_csv_reader().open_csv(csvPath)
        input_u.get_csv_reader().set_selected_column("system.u")
        input_u.set_covariance(2.0)

        # Set the CSV file associated to the output, and its covariance
        output = self.m.get_output_by_name("y")
        output.get_csv_reader().open_csv(csvPath)
        output.get_csv_reader().set_selected_column("system.y")
        output.set_measured_output()
        output.set_covariance(2.0)

    def set_state_to_estimate(self):
        """
        This method sets the state variable that needs to be estimated
        by the UKF. The state variable has also a constraints,
        it needs to be higher than zero.
        """
        # Select the states to be identified, and add it to the list
        self.m.add_variable(self.m.get_variable_object("x"))
    
        # Set initial value of state, and its covariance and the limits (if any)
        var = self.m.get_variables()[0]
        var.set_initial_value(1.5)
        var.set_covariance(0.5)
        var.set_min_value(0.0)
        var.set_constraint_low(True)
        
    def test_instantiate_UKF(self):
        """
        This method verifies the baility to correctly instantiate an 
        object of type :class:`estimationpy.ukf.ukf_fmu.UkfFmu`.
        """

        # Initialize the first order model
        self.set_first_order_model()

        # Instantiate the UKF for the FMU without state/parameters to estimate,
        # verify that raises an exception
        self.assertRaises(ValueError, UkfFmu, self.m, "The object initialized with a bad configured model should raise an exception")
        
        # Associate inputs and outputs
        self.set_first_order_model_input_outputs()
        # Define the variables to estimate
        self.set_state_to_estimate()

        # Retry to instantiate, now with a proper model
        ukf_FMU = UkfFmu(self.m) 

        # Get the parameters set by the initialization
        (alpha, beta, k, lambd, sqrt_C, N) = ukf_FMU.get_ukf_params()
                
        # Verify their default values
        self.assertEqual(1, N, "The number os states to estimate is not correct")
        self.assertEqual(alpha, 1.0/np.sqrt(3), "The base value for alpha is wrong")
        self.assertEqual(beta, 2, "The base value for beta is wrong")
        self.assertEqual(k, 3 - N, "The base value for k is wrong")
        
        # Get the parameters set by default function
        ukf_FMU.set_default_ukf_params()
        (alpha, beta, k, lambd, sqrt_C, N) = ukf_FMU.get_ukf_params()
                
        # Verify their default values
        self.assertEqual(alpha, 0.01, "The default value for alpha is wrong")
        self.assertEqual(beta, 2, "The default value for beta is wrong")
        self.assertEqual(k, 1, "The default value for k is wrong")
        
        # Compute and get the weights
        ukf_FMU.compute_weights()
        w_m, w_c = ukf_FMU.get_weights()

        # Verify the length
        self.assertEqual(len(w_c), 3, "Length of vector w_c is wrong")
        self.assertEqual(len(w_m), 3, "Length of vector w_m is wrong")
        
        # Verify that the first element of w_c is different from the first element of w_m
        self.assertTrue(w_c[0] != w_m[0], "The first elements of w_c and w_m must be different")

        # The remaining elements must be equal
        for i in range(1,N):
            self.assertEqual(w_c[i], w_m[i], "Weights w_m[{0}] and w_c[{0}] are different".format(i))
            
        # The sum of w_m is equal to 1
        self.assertEqual(np.sum(w_m), 1.0, "The weigts of w_m must sum up to 1, instead is {0}".format(np.sum(w_m)))
        
        return

    def test_vector_and_matrix_operations(self):
        """
        This method contains several checks for the methods provided by the class
        to operate with matrices and vectors.
        """

        # Initialize the first order model
        self.set_first_order_model()

        # Associate inputs and outputs
        self.set_first_order_model_input_outputs()

        # Define the variables to estimate
        self.set_state_to_estimate()

        # Retry to instantiate, now with a proper model
        ukf_FMU = UkfFmu(self.m)

        # Define square root matrix
        S = np.random.uniform(size=(6,6))
        S2 = np.dot(S, S.T)

        # Compute square matrix S
        s = ukf_FMU.square_root(S2)
        s2 = np.dot(s, s.T)

        np.testing.assert_almost_equal(s2, S2, 7, "The product of the square root matrix is not equal to the original S2")
        
        # Verify ability to apply constraints
        x = np.array([1.1])
        x_constr = ukf_FMU.constrained_state(x)
        self.assertEqual(x_constr, x, "This state vector doesn't require to be constrained")

        x = np.array([-1.1])
        x_constr = ukf_FMU.constrained_state(x)
        x_expected = np.zeros(1)
        self.assertEqual(x_constr, x_expected, "This state vector does require to be constrained and it's not")
        
        return

    def test_chol_update(self):
        """
        This method tests the Cholesky update method that is used to compute
        the squared root covariance matrix by the filter.
        """
        # Initialize the first order model
        self.set_first_order_model()

        # Associate inputs and outputs
        self.set_first_order_model_input_outputs()

        # Define the variables to estimate
        self.set_state_to_estimate()

        # Retry to instantiate, now with a proper model
        ukf_FMU = UkfFmu(self.m)
        
        # Number of points for computing the covariance matrix
        n = 100
        # Number of variables
        N = 300
        # True mean vector
        Xtrue = np.random.uniform(-8.0, 27.5, (1, N))
        
        # Generate the sample for computing the covariance matrix
        notUsed, N = Xtrue.shape
        Xpoints = np.zeros((n,N))
        for i in range(n):
	    noise = np.random.uniform(-2.0,2.0,(1,N)) 
	    Xpoints[i,:] = Xtrue + noise

        # default covariance to be added
        Q = 2.0*np.eye(N)

        # definition of the weights
        Weights = np.zeros(n)
        for i in range(n):
	    if i==0:
		Weights[i] = 0.5
	    else:
		Weights[i] = (1.0 - Weights[0])/np.float(n-1)

        #---------------------------------------------------
        # Standard method based on Cholesky
        i = 0
        P = Q
        for x in Xpoints:
	    error = x - Xtrue 
	    P     = P + Weights[i]*np.dot(error.T,error)
	    i    += 1
        S = ukf_FMU.square_root(P)
        
        np.testing.assert_almost_equal(P, np.dot(S, S.T), 8, \
                                       "Square root computed with basic Cholesky decomposition is not correct")

        #----------------------------------------------------
        # Test the Cholesky update
        sqrtQ = np.linalg.cholesky(Q)
        L = ukf_FMU.compute_S(Xpoints, Xtrue, sqrtQ, w = Weights)
        
        np.testing.assert_almost_equal(P, np.dot(L.T, L), 8, \
                                       "Square root computed with basic Cholesky update is not correct")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
