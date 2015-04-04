'''
Created on Jul 3, 2014

@author: marco
'''
import os
import unittest
import pyfmi
import numpy
import pandas as pd

from estimationpy.fmu_utils.in_out_var import InOutVar
from estimationpy.fmu_utils.csv_reader import CsvReader

class Test(unittest.TestCase):
    """
    This class contains unit tests for checking the behavior of the class
    
    FmuUtils.InOutVar
    
    that represents a wrapper around the class PyFmiVariable
    
    """

    def setUp(self):
        # Create an instance of the class InOutVar
        self.io_var = InOutVar()
        
        # Base path of this module, needed as reference path
        dirPath = os.path.dirname(__file__)
        # Paths of the files to be opened
        self.csvOK = os.path.join(dirPath, "resources", "simpleCSV.csv")
        
        # These are the values contained into the CSV file correct
        self.colNames = ["system.u", "system.x", "system.y"]
        self.t = numpy.linspace(0.0, 3.5, 8)
        self.u = numpy.array(1.0*numpy.ones(8))
        self.x = numpy.array(1.1*numpy.ones(8))
        self.y = numpy.array(1.3*numpy.ones(8))

    def tearDown(self):
        pass


    def test_read_value_in_FMU(self):
        """
        This function tests the method that reads a value in an FMU
        """
        pass
    
    def test_measured_output(self):
        """f
        This function tests the method that set and read if 
        an output variable is measured or not
        """
        
        self.io_var.set_measured_output(True)
        self.assertTrue(self.io_var.is_measured_output(), "The flag measured output has been set to True but value does not correspond")
        
        self.io_var.set_measured_output(False)
        self.assertFalse(self.io_var.is_measured_output(), "The flag measured output has been set to True but value does not correspond")
    
    def test_set_covariance(self):
        """
        This function tests the method SetCovariance
        """
        cov = -1.0
        self.assertFalse(self.io_var.set_covariance(cov), "The covariance cannot be set to a negative value")
        self.assertNotEqual(cov, self.io_var.get_covariance(), "The covariance shouldn't be set to the negative value")
        
        cov = 2.4
        self.assertTrue(self.io_var.set_covariance(cov), "The covariance should be set to the value")
        self.assertEqual(cov, self.io_var.get_covariance(), "The covariance should be set to the value")
    
    def test_set_object(self):
        """
        This function tests the method that associate a PyFmiVariable object to the object that is 
        part of the InOutVar class
        """
        
        # Check that if trying to set the wrong type of object an exception is raised
        self.assertRaises(TypeError, self.io_var.set_object, None)
        
        # Create a dummy pyfmi.fmi.Scalarvariable and try to associate it to the InOutVar object
        v = pyfmi.fmi.ScalarVariable(0, 0, 0)
        self.io_var.set_object(v)
        self.assertIsInstance(self.io_var.get_object(), pyfmi.fmi.ScalarVariable, "The object returned by the GetObject method does not return a pyfmi.fmi.ScalarVariable object")
        
        
    def test_set_csv_reader(self):
        """
        This function tests the method that associate a CsvReader object to the csv reader that is 
        part of the InOutVar class
        """
        
        # Check that if trying to set the wrong type of object an exception is raised
        self.assertRaises(TypeError, self.io_var.set_csv_reader, None)
        
        # Associate to it a real csv reader and try to read it
        reader = CsvReader()
        self.io_var.set_csv_reader(reader)
        self.assertIsInstance(self.io_var.get_csv_reader(), CsvReader, "The object returned by the GetCsvReader method does not return a CsvReader object")
    
    def test_get_data_series_from_csv(self):
        """
        This function tests the method that returns a data series once the csv file that contains it has
        been read
        """
        
        # Create and associate the csv reader
        reader = CsvReader()
        reader.open_csv(self.csvOK)
        self.io_var.set_csv_reader(reader)
        
        # Try to get a data series without specifying the selected column
        self.assertIsInstance(self.io_var.get_data_series(), pd.Series, "The reader has a CSV file assigned but not a column, the return value should be a pandas.Series")
        self.assertEqual(0, len(self.io_var.get_data_series()), "The reader has a CSV file assigned but not a column, the return value should be a pandas.Series empty")
        
        # Retrieve data and compare to known values
        col_name = self.colNames[0]
        data = pd.Series(self.u, index = pd.to_datetime(self.t, unit = "s"), name = col_name)
        
        # Select the column and retrieve the data 
        self.io_var.get_csv_reader().set_selected_column(col_name)
        # Read the value from the csv file associated
        self.assertTrue(self.io_var.read_data_series(), "The data series reda from the csv file is empty")
        
        # Now the dataSeries has been read, it is possible to get it
        self.assertTrue(numpy.allclose(data.values, self.io_var.get_data_series().values), "The pandas Series get is not equal to %s" % str(data.values))
        self.assertListEqual(data.index.tolist(), self.io_var.get_data_series().index.tolist(), "The index of the pandas Series get is not equal to %s" % str(data.index))
        
    def test_set_data_series(self):
        """
        This function tests the method that sets the dataseries stored in the InOutVar object.
        The dataseries has to be a pandas.Series object with a pandas.DatetimeIndex object as index.
        """
        
        # Trying to set wrong types raises an exception
        self.assertRaises(TypeError, self.io_var.set_data_series, None)
        self.assertRaises(TypeError, self.io_var.set_data_series, [])        
        self.assertRaises(TypeError, self.io_var.set_data_series, [1, 2, 3])
        
        # Trying to set a pandas.Series that has an integer index raises an exception
        s = pd.Series([1,2,3])
        self.assertRaises(TypeError, self.io_var.set_data_series, s)
        
        # Convert the index type to pandas.DatetimeIndex
        s.index = pd.to_datetime(s.index, unit="s")
        # Set the data series
        self.io_var.set_data_series(s)
        
        # Compare the values
        self.assertTrue(numpy.allclose(s.values, self.io_var.get_data_series().values), "The pandas Series get is not equal to %s" % str(s.values))
        self.assertListEqual(s.index.tolist(), self.io_var.get_data_series().index.tolist(), "The index of the pandas Series get is not equal to %s" % str(s.index))
    
    def test_read_from_data_series(self):
        """
        This function tests the method that allows to read a value from the data series associated to the 
        input/output variable
        """
        x = numpy.array([1, 2,  3,   5,  6,  7,  8, 16, 10])
        t = numpy.array([0, 10, 20, 30, 40, 50, 60, 80, 90])
        
        # Set a pandas.Series that has an integer index
        s = pd.Series(x, index = t)
        # Convert the index type to pandas.DatetimeIndex
        s.index = pd.to_datetime(s.index, unit="s")
        
        # Set the data series
        self.io_var.set_data_series(s)
        
        # Get the values at the specified points
        for ix in s.index:
            self.assertEqual(s[ix], self.io_var.read_from_data_series(ix), "The value read by the function ReadFromDataSeries does not correspond to the value set")
            
        # Read values at interpolated points
        new_ts = numpy.array([5.0, 22.0, 44.0, 70.0, 75.0, 5.0, 33.0, 12.0])
        
        for new_t in new_ts:
            # Create a new index 
            new_ix = pd.to_datetime(new_t, unit="s")
            
            # Compute value with method and with Numpy
            int_v = self.io_var.read_from_data_series(new_ix)
            num_int_v = numpy.interp(new_t, t, x)
            
            # Compare the results
            self.assertEqual(int_v, num_int_v, "The interpolated values do not match. Numpy = %.2f , ReadFromDataSeries = %.2f" % (num_int_v, int_v))
            
        # Check that returns False if the index is out of the range
        out_ts = numpy.array([-5.0, 90.1, -0.01])
        for out_t in out_ts:
            out_ix = pd.to_datetime(out_t, unit="s")    
            # Check that returns False
            self.assertFalse(self.io_var.read_from_data_series(out_ix), "The index is out of range and the method ReadFromDataSeries has to return False")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()