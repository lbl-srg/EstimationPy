'''
Created on Feb 25, 2014

@author: marco
'''
import unittest
import numpy

from FmuUtils import CsvReader

class Test(unittest.TestCase):


    def setUp(self):
        # Create an empty instance of the class CsvReader
        self.r = CsvReader.CsvReader()
        
        self.csvOK = "./Resources/SimpleCSV.csv"
        self.csvNotExisting = "thisFileDoesNotExist.csv"
        self.csvRepeated = "./Resources/SimpleCsvRepeatedValues.csv"
        self.csvUnsorted = "./Resources/SimpleCsvUnsortedValues.csv"
        
        # These are the values contained into the CSV file correct
        self.colNames = ["Time", "system.u", "system.x", "system.y"]
        self.t = numpy.linspace(0.0, 3.5, 8)
        self.u = numpy.matrix(1.0*numpy.ones(8))
        self.x = numpy.matrix(1.1*numpy.ones(8))
        self.y = numpy.matrix(1.3*numpy.ones(8))

    def tearDown(self):
        pass


    def test_instantiateCsvReader(self):
        # Performs some basic checks
        self.assertEqual("",self.r.GetFileName(), "The name of the file should be empty")
        self.assertListEqual([], self.r.GetColumnNames(), "Column names of the CsvReader have to be empty")
        self.assertFalse(self.r.SetSelectedColumn("columnName"), "Not possible to select a column with a missing csv file associated")
        self.assertEqual("", self.r.GetSelectedColumn(), "The selected column has to be an empty string")
    
    def test_loadCsvFile(self):
        
        # Try to open an existing Csv file
        self.assertTrue(self.r.OpenCSV(self.csvOK), "The file %s should be opened" % self.csvOK)
        self.assertListEqual(self.colNames, self.r.GetColumnNames(), "The column names are not the expected ones")
        
        for n in self.colNames:
            self.assertTrue(self.r.SetSelectedColumn(n),"The column named %s should be selected" % n)
            self.assertEqual(n, self.r.GetSelectedColumn(), "The column selected should be %s" % n)
        
        # try to open an not existing file
        self.assertFalse(self.r.OpenCSV(self.csvNotExisting), "The file %s does not exist and should not be opened" % self.csvNotExisting)
        
    def test_loadDataSeries(self):
        
        # Try to select a data series before assigning the file
        self.assertEqual({}, self.r.GetDataSeries(), "The reader has not a CSV file assigned, the return value should be {}")
        
        # Open an existing Csv file
        self.r.OpenCSV(self.csvOK)
        
        # Try to get a data series without specifying the selected column
        self.assertEqual({}, self.r.GetDataSeries(), "The reader has a file associated but is missing the selected column, the method should return {}")
        
        # Retrieve data and compare to known values
        data = {"data": self.t, "time": self.t}
        self.r.SetSelectedColumn(self.colNames[0])
        self.assertTrue(numpy.allclose(data["data"], self.r.GetDataSeries()["data"]), "The data series get is not equal to %s" % str(data))
        self.assertTrue(numpy.allclose(data["time"], self.r.GetDataSeries()["time"]), "The data series get is not equal to %s" % str(data))
        
        data = {"data": self.u, "time": self.t}
        self.r.SetSelectedColumn(self.colNames[1])
        self.assertTrue(numpy.allclose(data["data"], self.r.GetDataSeries()["data"]), "The data series get is not equal to %s" % str(data))
        self.assertTrue(numpy.allclose(data["time"], self.r.GetDataSeries()["time"]), "The data series get is not equal to %s" % str(data))
        self.assertFalse(numpy.allclose(data["data"], self.r.GetDataSeries()["time"]), "The data series have to be identified as different %s" % str(data))

        data = {"data": self.x, "time": self.t}
        self.r.SetSelectedColumn(self.colNames[2])
        self.assertTrue(numpy.allclose(data["data"], self.r.GetDataSeries()["data"]), "The data series get is not equal to %s" % str(data))
        self.assertTrue(numpy.allclose(data["time"], self.r.GetDataSeries()["time"]), "The data series get is not equal to %s" % str(data))
        self.assertFalse(numpy.allclose(data["data"], self.r.GetDataSeries()["time"]), "The data series have to be identified as different %s" % str(data))
        
        data = {"data": self.y, "time": self.t}
        self.r.SetSelectedColumn(self.colNames[3])
        self.assertTrue(numpy.allclose(data["data"], self.r.GetDataSeries()["data"]), "The data series get is not equal to %s" % str(data))
        self.assertTrue(numpy.allclose(data["time"], self.r.GetDataSeries()["time"]), "The data series get is not equal to %s" % str(data))
        self.assertFalse(numpy.allclose(data["data"], self.r.GetDataSeries()["time"]), "The data series have to be identified as different %s" % str(data))
        
    def test_RepeatedDataSeries(self):
        # Open an existing Csv file
        self.r.OpenCSV(self.csvRepeated)
        
        # Select one column
        self.r.SetSelectedColumn(self.colNames[1])
        
        # The time instants are not correct, it should return {}
        self.assertEqual({}, self.r.GetDataSeries(), "The reader has a file repeated time values, the method should return {}")
        
    def test_UnsortedDataSeries(self):
        # Open an existing Csv file
        self.r.OpenCSV(self.csvUnsorted)
        
        # Select one column
        self.r.SetSelectedColumn(self.colNames[1])
        
        # The time instants are not correct, it should return {}
        self.assertEqual({}, self.r.GetDataSeries(), "The reader has a file unsorted time values, the method should return {}")
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()