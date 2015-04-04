'''
Created on Sep 6, 2013

@author: marco
'''

import csv
import numpy
import pandas as pd

import Strings

class CsvReader():
    """
    
    This class provides the functionalities necessary to open and manage a csv file.
    This class is used because the input and output data series to be provided at the FMU are contained into
    csv files.
    
    The csv files should be like
    
    time, x, y, zeta, kappa hat, T [k]
    0.0, 3.4, 23.0, 22, 5, 77.5
    0.1, 3.4, 23.0, 22, 5, 77.3
    0.2, 3.4, 23.0, 22, 5, 76.8
    0.3, 3.4, 23.0, 22, 5, 34.4
    0.4, 3.4, 23.0, 22, 5, 72.22
    0.5, 3.4, 23.0, 22, 5, 71.9
    0.6, 3.4, 23.0, 22, 5, 70.9
    
    The first row IS ALWAYS the header of the file and contains a brief description of the columns.
    The other rows contain the data separated by commas.
    
    The first column HAS to be the time associated to the data series.
    This column will be used as index for the associated pandas data series. Since the first column is used
    as index its name will not appear between the column names.
    
    """
    
    def __init__(self, filename = ""):
        """
        Initialization method of the CsvReader class
        """
        
        # The default dialect is e
        self.dialect = csv.excel
        
        # file reference
        self.filename = filename
        
        # columns names
        self.columnNames = []
        
        # the identifier of the column selected in the CSV file
        self.columnSelected = None
    
    def __str__(self):
        """
        
        This method returns a string that describe the CSV reader object
        
        """
        string = "CsvReader Object"
        string += "\n-File: "+str(self.filename)
        string += "\n-Columns Available:"
        for c in self.columnNames:
            string += "\n\t-"+str(c)
        string += "\n-Selected: "+str(self.columnSelected)
        return string
    
    def __open_csv__(self, csv_file):
        """
        This method is used to open a csv file and create a corresponding pandas
        data frame
        """
        # Open the file passed as parameter.
        # Read the csv file and instantiate the data frame
        try:
            # Load the data frame
            df = pd.io.parsers.read_csv(self.filename, dialect = self.dialect)
                
            # Use the first column as index of the data frame
            df.set_index(df.columns[0], inplace = True, verify_integrity = True)
            
            # convert the index to a datetime object, assuming the values have been specified
            # using the SI unit for time [s]
            df.index = pd.to_datetime(df.index, unit="s")
            
            # Sort values with respect to the index
            df.sort(inplace = True)
            
            return df
        
        except IOError, e:
            print "The file %s does not exist, impossible to open " % self.filename
            print e
            return pd.DataFrame()
        
        except ValueError, e:
            print "The file %s has problem with the time index " % self.filename
            print e
            return pd.DataFrame()
     
    def open_csv(self, filename = ""):
        """
        
        Open a CSV file
        
        """
        # Reinitialize all
        self.__init__(filename)
        
        # Open the csv and get the Data frame
        df = self.__open_csv__(filename)
        
        # If the data frame is empty there were problems while loading the file
        if len(df.index) == 0:
            print "ERROR:: The csv file "+filename+" is not correct, please check it..."
            return False
        
        # Get the column names and then delete the data frame
        try:
            self.columnNames = df.columns.tolist()
            del(df)
            return True
        except csv.Error:
            print "ERROR:: The csv file "+filename+" is not correct, please check it..."
            del(df)
            return False
    
    def get_file_name(self):
        """
        This method returns the filename of the CSV file associated
        """
        return self.filename
        
    def get_column_names(self):
        """
        This method returns a list containing the names of the columns contained in the csv file
        """
        return self.columnNames
    
    def set_selected_column(self, columnName):
        """
        This method allows to specify which is the column to be selected
        """
        if columnName in self.get_column_names():
            self.columnSelected = columnName
            return True
        else:
            print "ERROR:: The column selected",str(columnName),"is not part of the columns names list",str(self.columnNames)
            return False
            
    def get_selected_column(self):
        """
        This method returns the column selected for this CSV reader
        """
        if self.columnSelected != None:
            return self.columnSelected
        else:
            return ""
            
    def print_dialect_information(self):
        """
        This method print the information about the dialect used by the Csv Reader
        """
        print "CsvReader Dialect informations:"
        print "* Delimiter: "+str(self.dialect.delimiter)
        print "* Double quote char: "+str(self.dialect.doublequote)
        print "* Escape char: "+str(self.dialect.escapechar)
        print "* Skip initial space: "+str(self.dialect.skipinitialspace)
        print "* Quoting char: "+str(self.dialect.quoting)
        print "* Line terminator: "+str(self.dialect.lineterminator)
        
    def get_data_series(self):
        """
        Once the csv file and its column have been selected, it is possible to read the data from the csv
        file and return them. Please remember that the first column of the csv file HAS to be the time.
        This method returns a pandas data series associated to the selected column.
        """
        # initialize with empty pandas data series
        dataSeries = pd.Series()
        
        # Check if the file name has been selected
        if self.filename != None and self.filename != "":
            
            # Open the csv and get the Data frame
            df = self.__open_csv__(self.filename)
            
            # If the data frame is empty there were problems while loading the file
            if len(df.index) == 0:
                print "ERROR:: The csv file "+self.filename+" is not correct, please check it..."
                return dataSeries
            
            # Check if the column name is set
            if self.columnSelected != None:
                
                # Check if the column name is part of the available dictionary
                if self.columnSelected in self.columnNames:
                    
                    # Read the time and data column from the csv file
                    dataSeries = df[self.columnSelected]
                        
                    return dataSeries  
                    
                else:
                    print "The column selected must be present in the csv file!"
                    print "Column selected: ", self.columnSelected
                    print "Columns available: ", self.columnNames
                    return dataSeries
            else:
                print "Select a column for the csv file!"
                return dataSeries
        else:
            print "Select a file for the CSV before trying to read it!"
            return dataSeries