'''
@author: Marco Bonvini
'''

import csv
import numpy
import pandas as pd

from estimationpy.fmu_utils import strings

class CsvReader():
    """
    
    This class provides functionalities that can be used to provide input to an
    FMU model or to a state/parameter estimation algorithm through CSV files.
    
    An object of class :class:`CsvReader` manages CSV file that have the
    following format::
    
        time, x, y, zeta, kappa hat, T [k]
        0.0, 3.4, 23.0, 22, 5, 77.5
        0.1, 3.4, 23.0, 22, 5, 77.3
        0.2, 3.4, 23.0, 22, 5, 76.8
        0.3, 3.4, 23.0, 22, 5, 34.4
        0.4, 3.4, 23.0, 22, 5, 72.22
        0.5, 3.4, 23.0, 22, 5, 71.9
        0.6, 3.4, 23.0, 22, 5, 70.9
    
    The first row is mandatory and is the header of the corresponding table created
    when importing the CSV file. The remaining rows contain data that are separated by
    commas.
    
    The first column is time associated to the data series.
    This column will be used as index to create a pandas DataFrame.
    Since the first column is used as index, its name won't be available among
    the column names associated tot the DataFrame.
    
    """
    
    def __init__(self, filename = ""):
        """
        Constructor for the class :class:`CsvReader`.
        The method initializes the type of dialect used to interpet the CSV file,
        the list containing the names of the columns, and which columns
        are selected.
        
        :param str filename: The path that defines the CSV file to open.
           The argument is optinal because it can be specified later.
        
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
        This method returns a string representation of the
        instance.
        
        :return: a String representation of the instance.
        :rtype: string
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
        This private method is used to open a CSV file given a path name specified by the
        parameter ``csv_file``.
        The method uses the function ``pandas.io.parsers.read_csv`` to open
        the file.
        
        :param str filename: The path that defines the CSV file to open.
        
        :return: The DataFrame object containing the data of the CSV file.
        :rtype: pandas.DataFrame
        
        :raises IOError: The method raises an ``IOErrror`` if the file specified by the argument
            ``csv_file`` does not exist.
            
        :raises ValueError: The method raises an ``ValueError`` if the file specified by the argument
            ``csv_file`` can't be succesfully indexed.
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
     
    def open_csv(self, filename):
        """
        This method open a CSV file given a path name specified by the
        parameter ``filename``.
        The method uses the underlying method :func:`__open_csv__` to open
        the file.
        
        :param str filename: The path that defines the CSV file to open.
        
        :return: True is the CSV is loaded and is not empty, False otherwise.
        :rtype: bool
        
        :raises IOError: The method raises an ``IOErrror`` if the file specified by the argument
            ``csv_file`` does not exist.
        
        :raises ValueError: The method raises an ``ValueError`` if the file specified by the argument
            ``csv_file`` can't be succesfully indexed.
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
        This method returns the filename of the CSV file associated to this object.
        
        :return: a String representing the path of the CSV file.
        :rtype: string
        """
        return self.filename
        
    def get_column_names(self):
        """
        This method returns a list containing the names of the columns contained in the csv file.
        
        :return: a List containing the names of the columns in the CSV file.
        :rtype: list(str)
        
        """
        return self.columnNames
    
    def set_selected_column(self, columnName):
        """
        This method allows to specify which of the columns in the CSV file is selected.
        Once a column is selecetd, it's possible to get the corresponding ``pandas.Series``
        with the method :func:`get_data_series` .
        
        :param str columnName: The name of the column to be selected.
        
        :return: True if the name is successfully selected, False otherwise (e.g., if
            the name is not present in the available column names).
        :rtype: bool
        """
        if columnName in self.get_column_names():
            self.columnSelected = columnName
            return True
        else:
            print "ERROR:: The column selected",str(columnName),"is not part of the columns names list",str(self.columnNames)
            return False
            
    def get_selected_column(self):
        """
        This method returns the name of the column selected.
        
        :return: Name of the column selected. If no columns are selected the
          method returns an empty string.
        :rtype: string
        
        """
        if self.columnSelected != None:
            return self.columnSelected
        else:
            return ""
            
    def print_dialect_information(self):
        """
        This method print the information about the dialect used by the Csv Reader.
        
        :return: None
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
        This method returns a pandas Series object that contains the data 
        read from the CSV file at the column selected with the method :func:`set_selected_column`.
        
        Before calling this method make sure:
        
        1. the path of the CSV file has been specified,
        2. the name of the column to selected has been specified.
        
        :return: A Series object representing the time series data contained in the selected column.
            In case the file is not specified, or the column is not specified, the method
            returns an empty Series.
            
        :rtype: pandas.Series
        
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
