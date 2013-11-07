'''
Created on Nov 6, 2013

@author: marco
'''
from CsvReader import CsvReader

class InOutVar():
    """
    
    This class either represent an input or output variable.
    Both are variable with associated a data series contained in a .csv file
    
    """
    
    def __init__(self, object = None):
        """
        Initialization method
        """
        self.object = object
        self.CsvReader = CsvReader()
        
    def SetObject(self, object):
        """
        Set the object <<pyfmi.ScalarVariable>> associated to the input/output
        """
        self.object = object
        
    def GetObject(self):
        """
        Get the object <<pyfmi.ScalarVariable>> associated to the input/output
        """
        return self.object
    
    def SetCsvReader(self, reader):
        """
        Set the CsvReader class associated to the input/output
        """
        self.CsvReader = reader
        
    def GetCsvReader(self):
        """
        Get the CsvReader class associated to the input/output
        """
        return self.CsvReader