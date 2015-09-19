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

        return

    def set_valve_model(self):
        """
        This method is used in different tests and it
        loads the model of the faulty valve.
        """
        # Define the path of the FMU file
        if platform.architecture()[0]=="32bit":
            filePath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "FMUs", "ValveStuck.fmu")
        else:
            filePath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "FMUs", "ValveStuck.fmu")
        
        # Initialize the FMU model empty
        self.m = Model(filePath)

        return

    def set_first_order_model_input_outputs(self, noisy = True):
        """
        This methos associates the input and the measured output of
        the first order model from a CSV file.
        If the boolean flag ``noisy`` is equal to True then the data that is
        loaded contains noise, otherwise it is clean.
        
        :param bool noisy: flag that indicates which type of data to load.
        """
        # Path of the csv file containing the data series
        if noisy:
            csvPath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "data", "NoisySimulationData_FirstOrder.csv")
        else:
            csvPath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")
            
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

        return

    def set_valve_model_input_outputs(self):
        """
        This method associates the input and measured outputs of the
        valve model from a CSV file.
        """
        # Path of the csv file containing the data series
        csvPath = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "data", "NoisyData_ValveStuck.csv")

        # Set the CSV file associated to the input, and its covariance
        input = self.m.get_input_by_name("dp")
        input.get_csv_reader().open_csv(csvPath)
        input.get_csv_reader().set_selected_column("valveStuck.dp")
        
        # Set the CSV file associated to the input, and its covariance
        input = self.m.get_input_by_name("cmd")
        input.get_csv_reader().open_csv(csvPath)
        input.get_csv_reader().set_selected_column("valveStuck.cmd")
        
        # Set the CSV file associated to the input, and its covariance
        input = self.m.get_input_by_name("T_in")
        input.get_csv_reader().open_csv(csvPath)
        input.get_csv_reader().set_selected_column("valveStuck.T_in")
    
        # Set the CSV file associated to the output, and its covariance
        output = self.m.get_output_by_name("m_flow")
        output.get_csv_reader().open_csv(csvPath)
        output.get_csv_reader().set_selected_column("valveStuck.m_flow")
        output.set_measured_output()
        output.set_covariance(0.05)

        return
    
    def set_state_to_estimate_first_order(self):
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

        return

    def set_state_and_param_to_estimate_valve(self):
        """
        This method sets the state variable that needs to be estimated
        by the UKF and the smoother in the valve example. 
        The state variable and the parameter have both constraints.
        The valve position needs to be between [0,1] while the thermal drift
        coefficient has to be between [-0.005, and 0.025].
        """
        # Select the variable to be estimated
        self.m.add_variable(self.m.get_variable_object("command.y"))
        
        # Set initial value of parameter, and its covariance and the limits (if any)
        var = self.m.get_variables()[0]
        var.set_initial_value(1.0)
        var.set_covariance(0.05)
        var.set_min_value(0.0)
        var.set_constraint_low(True)
        var.set_max_value(1.00)
        var.set_constraint_high(True)
        
        #################################################################
        # Select the parameter to be estimated
        self.m.add_parameter(self.m.get_variable_object("lambda"))
        
        # Set initial value of parameter, and its covariance and the limits (if any)
        var = self.m.get_parameters()[0]
        var.set_initial_value(0.00)
        var.set_covariance(0.0007)
        var.set_min_value(-0.005)
        var.set_constraint_low(True)
        var.set_max_value(0.025)
        var.set_constraint_high(True)

        return
    
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
        self.set_state_to_estimate_first_order()

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
        self.set_state_to_estimate_first_order()

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
        self.set_state_to_estimate_first_order()

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

        return

    def test_create_sigma_points(self):
        """
        This method tests the Cholesky update method that is used to compute
        the squared root covariance matrix by the filter.
        """
        # Initialize the first order model
        self.set_first_order_model()

        # Associate inputs and outputs
        self.set_first_order_model_input_outputs()

        # Define the variables to estimate
        self.set_state_to_estimate_first_order()
        
        # Instantiate with a proper model
        ukf_FMU = UkfFmu(self.m)

        # Verify that the method raises an exception if the
        # inputs are wrong
        self.assertRaises(ValueError, ukf_FMU.compute_sigma_points, np.zeros(3), np.zeros(3), np.zeros((3,3)))
        self.assertRaises(ValueError, ukf_FMU.compute_sigma_points, np.zeros(2), np.zeros(3), np.zeros((3,3)))
        self.assertRaises(ValueError, ukf_FMU.compute_sigma_points, np.zeros(1), np.array([]), np.zeros((3,3)))

        # Create the sigma points
        x0 = np.array([2.5])
        sigma_points = ukf_FMU.compute_sigma_points(x0, np.array([]), np.diag(np.ones(1)))

        # Check the size
        self.assertTrue(sigma_points.shape == (3,1), "The size of the sigma points is not correct")
        
        # Verify that the first sigma point is equal to the center
        self.assertEqual(x0, sigma_points[0,:], "First sigma point is not [0]")

        # Verify that the second and the last sigma points are symmetric
        self.assertEqual(0.5*(sigma_points[1,:] + sigma_points[2,:]), x0, "The sigma points 1,2 are not symmetric with respect to 0")
        
        return

    def test_project_sigma_points(self):
        """
        This method tests the function that projects the sigma points
        by running a simulation.
        """

        # Initialize the first order model
        self.set_first_order_model()

        # Associate inputs and outputs
        self.set_first_order_model_input_outputs(noisy = False)

        # Define the variables to estimate
        self.set_state_to_estimate_first_order()

        # Initialize the simulator
        self.m.initialize_simulator()
        
        # Instantiate with a proper model
        ukf_FMU = UkfFmu(self.m)

        # Define the sigma points
        x0 = np.array([2.5])
        sigma_points = ukf_FMU.compute_sigma_points(x0, np.array([]), np.diag(np.ones(1)))

        # Propagate the points by simulating from 0 to 14.5 seconds
        t0 = pd.to_datetime(0.0, unit = "s", utc = True)
        t1 = pd.to_datetime(14.5, unit = "s", utc = True)
        X_proj, Z_proj, Xfull_proj, Zfull_proj = ukf_FMU.sigma_point_proj(sigma_points, t0, t1)

        # Verify that they started from different initial coditions and that they converged
        # at the same value after 12 seconds
        np.testing.assert_almost_equal(X_proj, 2.5*np.ones((3,1)), 3, "Verify that the solutions all converge to 2.5")

        # Compute their average using the method provided by the object and verify its value
        x_avg = ukf_FMU.average_proj(X_proj)

        np.testing.assert_almost_equal(x_avg, np.array([[2.5]]), 4, "Average of the propagated points is not correct")
        
        return
    
    def test_ukf_filter_first_order(self):
        """
        This method tests the ability of the filter to estimate the state
        of the first order system.
        """
        # Initialize the first order model
        self.set_first_order_model()

        # Associate inputs and outputs
        self.set_first_order_model_input_outputs()

        # Define the variables to estimate
        self.set_state_to_estimate_first_order()

        # Initialize the simulator
        self.m.initialize_simulator()

        # Retry to instantiate, now with a proper model
        ukf_FMU = UkfFmu(self.m)

        # Start the filter
        t0 = pd.to_datetime(0.0, unit = "s", utc = True)
        t1 = pd.to_datetime(30.0, unit = "s", utc = True)
        time, x, sqrtP, y, Sy, y_full = ukf_FMU.filter(start = t0, stop = t1, verbose = False)

        # Convert the results to numpy array
        time = time - time[0]
        time = np.array(map(lambda x: x.total_seconds(), time))
        x = np.array(x)
        y = np.array(y)
        sqrtP = np.array(sqrtP)
        Sy = np.array(Sy)
        y_full = np.squeeze(np.array(y_full))
        
        # Path of the csv file containing the True data series
        path_csv_simulation = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_FirstOrder.csv")

        # Compare the estimated states with the ones used to generate the data
        df_sim = pd.read_csv(path_csv_simulation, index_col = 0)
        time_sim = df_sim.index.values
        
        # Difference between state estimated and real state
        x_sim = np.interp(time, time_sim, df_sim["system.x"])
        err_state = np.abs(x_sim - x[:,0])

        # Identify maximum error and the time when it occurs
        max_error = np.max(err_state)
        t_max_error = np.where(err_state == max_error)

        # Make sure that the maximum error is less or equal than 0.5, and it happens at
        # the first time instant t = 0
        self.assertTrue(max_error <= 0.5, "The maximum error in the estimation has to be less than 0.5")
        self.assertTrue(t_max_error[0][0] == 0.0 and len(t_max_error[0]) == 1,\
                      "The maximum error is one and it is at t = 0")
        
        # Compute the mean absolute error
        avg_error = np.mean(err_state)
        self.assertTrue(avg_error < 0.06, "The average error should be less than 0.06")
        
        # Compute that the estimation +/- covariance contains the real state
        x_plus_sigma = x[:,0] + sqrtP[:,0,0]
        x_minus_sigma = x[:,0] - sqrtP[:,0,0]
        self.assertTrue(len(np.where(x_sim < x_minus_sigma)[0]) == 0,\
                        "The state estimation must contain the real state in its boundaries")
        self.assertTrue(len(np.where(x_sim > x_plus_sigma)[0]) == 0,\
                        "The state estimation must contain the real state in its boundaries")

        return

    def test_ukf_smoother_valve(self):
        """
        This method tests the state and parameter estimation on the valve example performed
        with the UKF + Smoother.
        """
        # Initialize the first order model
        self.set_valve_model()

        # Associate inputs and outputs
        self.set_valve_model_input_outputs()

        # Define the variables to estimate
        self.set_state_and_param_to_estimate_valve()

        # Initialize the simulator
        self.m.initialize_simulator()

        # Set models parameters
        use_cmd = self.m.get_variable_object("use_cmd")
        self.m.set_real(use_cmd, 0.0)

        lambd = self.m.get_variable_object("lambda")
        self.m.set_real(lambd, 0.0)
        
        # Retry to instantiate, now with a proper model
        ukf_FMU = UkfFmu(self.m)

        # Start filter and smoother
        t0 = pd.to_datetime(0.0, unit = "s", utc = True)
        t1 = pd.to_datetime(360.0, unit = "s", utc = True)
        time, x, sqrtP, y, Sy, y_full, Xsmooth, Ssmooth, Yfull_smooth = \
        ukf_FMU.filter_and_smooth(start = t0, stop = t1, verbose = False)

        # Convert the results to numpy array
        time = time - time[0]
        time = np.array(map(lambda x: x.total_seconds(), time))
        x = np.array(x)
        y = np.array(y)
        sqrtP = np.array(sqrtP)
        Sy = np.array(Sy)
        y_full = np.squeeze(np.array(y_full))
        xs = np.array(Xsmooth)
        Ss = np.array(Ssmooth)
        Ys = np.array(Yfull_smooth)
        
        # Path of the csv file containing the True data series generated by a simulation model
        path_csv_simulation = os.path.join(dir_path, "..", "modelica", "FmuExamples", "Resources", "data", "SimulationData_ValveStuck.csv")

        # Compare the estimated states with the ones used to generate the data
        df_sim = pd.read_csv(path_csv_simulation, index_col = 0)
        time_sim = df_sim.index.values

        # Difference between state estimated and real state/parameters
        opening_sim = np.interp(time, time_sim, df_sim["valveStuck.valve.opening"])
        lambda_sim = np.interp(time, time_sim, df_sim["valveStuck.lambda"])
        
        err_opening = np.abs(opening_sim - x[:,0])
        err_lambda = np.abs(lambda_sim - x[:,1])
        err_opening_s = np.abs(opening_sim - xs[:,0])
        err_lambda_s = np.abs(lambda_sim - xs[:,1])

        # Compute the maximum errors for both filter and smoother
        max_opening_error = np.max(err_opening)
        max_opening_error_s = np.max(err_opening_s)
        max_lambda_error = np.max(err_lambda)
        max_lambda_error_s = np.max(err_lambda_s)

        # Compute average error for both filter and smoother
        avg_opening_error = np.mean(err_opening)
        avg_opening_error_s = np.mean(err_opening_s)
        avg_lambda_error = np.mean(err_lambda)
        avg_lambda_error_s = np.mean(err_lambda_s)

        # Compare performances of UKF and Smoother, verify that the smoother improves
        # the estimation
        self.assertTrue(max_opening_error >= max_opening_error_s,\
                        "The max error in the estimation of the opening by the smoother is larger than the filter")
        self.assertTrue(max_lambda_error >= max_lambda_error_s,\
                        "The maxerror in the estimation of the drift coeff. by the smoother is larger than the filter")
        self.assertTrue(avg_opening_error >= avg_opening_error_s,\
                        "The avg error in the estimation of the opening by the smoother is larger than the filter")
        self.assertTrue(avg_lambda_error >= avg_lambda_error_s,\
                        "The avg error in the estimation of the drift coeff. by the smoother is larger than the filter")

        # Verify that some absolute perfomances are guaranteed
        self.assertTrue(0.08 > max_opening_error, "The maximum error of the UKF on the opening is too big")
        self.assertTrue(0.065 > max_opening_error_s, "The maximum error of the Smoother on the opening is too big")
        self.assertTrue(0.0117 > avg_opening_error, "The average error of the UKF on the opening is too big")
        self.assertTrue(0.0088 > avg_opening_error_s, "The average error of the Smoother on the opening is too big")

        self.assertTrue(0.0101 > max_lambda_error, "The maximum error of the UKF on the drift coef. is too big")
        self.assertTrue(0.0096 > max_lambda_error_s, "The maximum error of the Smoother on the drift coef. is too big")
        self.assertTrue(0.0028 > avg_lambda_error, "The average error of the UKF on the drift coef. is too big")
        self.assertTrue(0.00144 > avg_lambda_error_s, "The average error of the Smoother on the drift coef. is too big")
        
        return
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
