'''
Created on Feb 25, 2014

@author: marco
'''
import unittest
import numpy
import os
import pandas as pd

from estimationpy.fmu_utils import csv_reader

import logging
from estimationpy.fmu_utils import estimationpy_logging
estimationpy_logging.configure_logger(log_level = logging.DEBUG, log_level_console = logging.INFO, log_level_file = logging.DEBUG)


class Test(unittest.TestCase):

    def setUp(self):
        # Create an empty instance of the class CsvReader
        self.r = csv_reader.CsvReader()
        
        # Base path of this module, needed as reference path
        dirPath = os.path.dirname(__file__)
        
        # Paths of the files to be opened
        self.csvOK = os.path.join(dirPath, "resources", "simpleCSV.csv")
        self.csvNotExisting = os.path.join(dirPath, "resources", "thisFileDoesNotExist.csv")
        self.csvRepeated = os.path.join(dirPath, "resources", "simpleCsvRepeatedValues.csv")
        self.csvUnsorted = os.path.join(dirPath, "resources", "simpleCsvUnsortedValues.csv")
        
        # These are the values contained into the CSV file correct
        self.colNames = ["system.u", "system.x", "system.y"]
        self.t = numpy.linspace(0.0, 3.5, 8)
        self.u = numpy.array(1.0*numpy.ones(8))
        self.x = numpy.array(1.1*numpy.ones(8))
        self.y = numpy.array(1.3*numpy.ones(8))
        
        # These are the values contained in the unsorted file
        self.t_unsorted = numpy.array([0.0, 0.5, 0.9, 1.0, 1.6, 2.0, 2.5, 3.0, 3.5])
        self.u_unsorted = numpy.array(numpy.arange(1.0, 10.0))
        
    def tearDown(self):
        pass


    def test_instantiate_csv_reader(self):
        # Performs some basic checks
        self.assertEqual("",self.r.get_file_name(), "The name of the file should be empty")
        self.assertListEqual([], self.r.get_column_names(), "Column names of the CsvReader have to be empty")
        self.assertFalse(self.r.set_selected_column("columnName"), "Not possible to select a column with a missing csv file associated")
        self.assertEqual("", self.r.get_selected_column(), "The selected column has to be an empty string")
    
    def test_load_csv_file(self):
        # Try to open an existing Csv file
        self.assertTrue(self.r.open_csv(self.csvOK), "The file %s should be opened" % self.csvOK)
        self.assertListEqual(self.colNames, self.r.get_column_names(), "The column names %s are not the expected ones %s" % (str(self.r.get_column_names()), str(self.colNames)))
        
        for n in self.colNames:
            self.assertTrue(self.r.set_selected_column(n),"The column named %s should be selected" % n)
            self.assertEqual(n, self.r.get_selected_column(), "The column selected should be %s" % n)
        
        # try to open an not existing file
        self.assertFalse(self.r.open_csv(self.csvNotExisting), "The file %s does not exist and should not be opened" % self.csvNotExisting)
        
    def test_load_data_series(self):
        
        # Try to select a data series before assigning the file
        self.assertIsInstance(self.r.get_data_series(), pd.Series, "The reader has not a CSV file assigned, the return value should be a pandas.Series")
        self.assertEqual(0, len(self.r.get_data_series()), "The reader has not a CSV file assigned, the return value should be a pandas.Series empty")
        
        # Open an existing Csv file
        self.r.open_csv(self.csvOK)
        
        # Try to get a data series without specifying the selected column
        self.assertIsInstance(self.r.get_data_series(), pd.Series, "The reader has a CSV file assigned but not a column, the return value should be a pandas.Series")
        self.assertEqual(0, len(self.r.get_data_series()), "The reader has a CSV file assigned but not a column, the return value should be a pandas.Series empty")
        
        # Retrieve data and compare to known values
        col_name = self.colNames[0]
        data = pd.Series(self.u, index = pd.to_datetime(self.t, unit = "s", utc = True), name = col_name)
                 
        self.r.set_selected_column(col_name)
        self.assertTrue(numpy.allclose(data.values, self.r.get_data_series().values), "The pandas Series get is not equal to %s" % str(data.values))
        self.assertListEqual(data.index.tolist(), self.r.get_data_series().index.tolist(), "The index of the pandas Series get is not equal to %s" % str(data.index))
        
        # Retrieve data and compare to known values
        col_name = self.colNames[1]
        data = pd.Series(self.x, index = pd.to_datetime(self.t, unit = "s", utc = True), name = col_name)
                 
        self.r.set_selected_column(col_name)
        self.assertTrue(numpy.allclose(data.values, self.r.get_data_series().values), "The pandas Series get is not equal to %s" % str(data.values))
        self.assertListEqual(data.index.tolist(), self.r.get_data_series().index.tolist(), "The index of the pandas Series get is not equal to %s" % str(data.index))
        
        # Retrieve data and compare to known values
        col_name = self.colNames[2]
        data = pd.Series(self.y, index = pd.to_datetime(self.t, unit = "s", utc = True), name = col_name)
                 
        self.r.set_selected_column(col_name)
        self.assertTrue(numpy.allclose(data.values, self.r.get_data_series().values), "The pandas Series get is not equal to %s" % str(data.values))
        self.assertListEqual(data.index.tolist(), self.r.get_data_series().index.tolist(), "The index of the pandas Series get is not equal to %s" % str(data.index))
        
    def test_repeated_data_series(self):
        # Open an existing Csv file
        self.r.open_csv(self.csvRepeated)
        
        # Select one column, the first one
        self.r.set_selected_column(self.colNames[0])
        
        # The time instants are not correct, it should return pands Series that is empty
        self.assertIsInstance(self.r.get_data_series(), pd.Series, "The reader has a file repeated time values, the method should return a pandas.Series")
        self.assertEqual(0, len(self.r.get_data_series()), "The reader has a file repeated time values, the method should return a pandas.Series empty")
        
    def test_unsorted_data_series(self):
        # Open an existing Csv file
        self.r.open_csv(self.csvUnsorted)
        
        # Retrieve data unsorted in the csv. The data should be automatically sorted and then
        # compared to the to known values
        col_name = self.colNames[0]
        data = pd.Series(self.u_unsorted, index = pd.to_datetime(self.t_unsorted, unit = "s", utc = True), name = col_name)
                 
        self.r.set_selected_column(col_name)
        self.assertTrue(numpy.allclose(data.values, self.r.get_data_series().values), "The pandas Series get is not equal to %s" % str(data.values))
        self.assertListEqual(data.index.tolist(), self.r.get_data_series().index.tolist(), "The index of the pandas Series get is not equal to %s" % str(data.index))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()