'''
Created on Feb 25, 2014

@author: marco
'''
import unittest
import numpy
import os

from FmuUtils import CsvReader

class Test(unittest.TestCase):


    def setUp(self):
        # Create an empty instance of the class CsvReader
        self.r = CsvReader.CsvReader()
        
        # Base path of this module, needed as reference path
        dirPath = os.path.dirname(__file__)
        # Paths of the files to be opened
        self.csvOK = os.path.join(dirPath, "Resources", "SimpleCSV.csv")
        self.csvNotExisting = os.path.join(dirPath, "Resources", "thisFileDoesNotExist.csv")
        self.csvRepeated = os.path.join(dirPath, "Resources", "SimpleCsvRepeatedValues.csv")
        self.csvUnsorted = os.path.join(dirPath, "Resources", "SimpleCsvUnsortedValues.csv")
        
        # These are the values contained into the CSV file correct
        self.colNames = ["system.u", "system.x", "system.y"]
        self.t = numpy.linspace(0.0, 3.5, 8)
        self.u = numpy.matrix(1.0*numpy.ones(8))
        self.x = numpy.matrix(1.1*numpy.ones(8))
        self.y = numpy.matrix(1.3*numpy.ones(8))
        
        # These are the values contained in the unsorted file
        self.u_unsorted = numpy.matrix(numpy.arange(1.0, 10.0))
        self.t_unsorted = numpy.array([0.0, 0.5, 0.9, 1.0, 1.6, 2.0, 2.5, 3.0, 3.5])
        
    def tearDown(self):
        pass


    def Rest_instantiateCsvReader(self):
        # Performs some basic checks
        self.assertEqual("",self.r.GetFileName(), "The name of the file should be empty")
        self.assertListEqual([], self.r.GetColumnNames(), "Column names of the CsvReader have to be empty")
        self.assertFalse(self.r.SetSelectedColumn("columnName"), "Not possible to select a column with a missing csv file associated")
        self.assertEqual("", self.r.GetSelectedColumn(), "The selected column has to be an empty string")
    
    def Rest_loadCsvFile(self):
        # Try to open an existing Csv file
        self.assertTrue(self.r.OpenCSV(self.csvOK), "The file %s should be opened" % self.csvOK)
        self.assertListEqual(self.colNames, self.r.GetColumnNames(), "The column names %s are not the expected ones %s" % (str(self.r.GetColumnNames()), str(self.colNames)))
        
        for n in self.colNames:
            self.assertTrue(self.r.SetSelectedColumn(n),"The column named %s should be selected" % n)
            self.assertEqual(n, self.r.GetSelectedColumn(), "The column selected should be %s" % n)
        
        # try to open an not existing file
        self.assertFalse(self.r.OpenCSV(self.csvNotExisting), "The file %s does not exist and should not be opened" % self.csvNotExisting)
        
    def Rest_loadDataSeries(self):
        
        # Try to select a data series before assigning the file
        self.assertEqual({}, self.r.GetDataSeries(), "The reader has not a CSV file assigned, the return value should be {}")
        
        # Open an existing Csv file
        self.r.OpenCSV(self.csvOK)
        
        # Try to get a data series without specifying the selected column
        self.assertEqual({}, self.r.GetDataSeries(), "The reader has a file associated but is missing the selected column, the method should return {}")
        
        # Retrieve data and compare to known values
        data = {"data": self.u, "time": self.t}
        self.r.SetSelectedColumn(self.colNames[0])
        self.assertTrue(numpy.allclose(data["data"], self.r.GetDataSeries()["data"]), "The data series get is not equal to %s" % str(data))
        self.assertTrue(numpy.allclose(data["time"], self.r.GetDataSeries()["time"]), "The data series get is not equal to %s" % str(data))
        self.assertFalse(numpy.allclose(data["data"], self.r.GetDataSeries()["time"]), "The data series have to be identified as different %s" % str(data))

        data = {"data": self.x, "time": self.t}
        self.r.SetSelectedColumn(self.colNames[1])
        self.assertTrue(numpy.allclose(data["data"], self.r.GetDataSeries()["data"]), "The data series get is not equal to %s" % str(data))
        self.assertTrue(numpy.allclose(data["time"], self.r.GetDataSeries()["time"]), "The data series get is not equal to %s" % str(data))
        self.assertFalse(numpy.allclose(data["data"], self.r.GetDataSeries()["time"]), "The data series have to be identified as different %s" % str(data))
        
        data = {"data": self.y, "time": self.t}
        self.r.SetSelectedColumn(self.colNames[2])
        self.assertTrue(numpy.allclose(data["data"], self.r.GetDataSeries()["data"]), "The data series get is not equal to %s" % str(data))
        self.assertTrue(numpy.allclose(data["time"], self.r.GetDataSeries()["time"]), "The data series get is not equal to %s" % str(data))
        self.assertFalse(numpy.allclose(data["data"], self.r.GetDataSeries()["time"]), "The data series have to be identified as different %s" % str(data))
        
    def test_RepeatedDataSeries(self):
        # Open an existing Csv file
        self.r.OpenCSV(self.csvRepeated)
        
        # Select one column
        self.r.SetSelectedColumn(self.colNames[0])
        
        # The time instants are not correct, it should return {}
        self.assertEqual({}, self.r.GetDataSeries(), "The reader has a file repeated time values, the method should return {}")
        
    def test_UnsortedDataSeries(self):
        # Open an existing Csv file
        self.r.OpenCSV(self.csvUnsorted)
        
        # Retrieve data and compare to known values
        data = {"data": self.u_unsorted, "time": self.t_unsorted}
        self.r.SetSelectedColumn(self.colNames[0])
        self.assertTrue(numpy.allclose(data["data"], self.r.GetDataSeries()["data"]), "The data series get is not equal to %s" % str(data))
        self.assertTrue(numpy.allclose(data["time"], self.r.GetDataSeries()["time"]), "The data series get is not equal to %s" % str(data))
        self.assertFalse(numpy.allclose(data["data"], self.r.GetDataSeries()["time"]), "The data series have to be identified as different %s" % str(data))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()