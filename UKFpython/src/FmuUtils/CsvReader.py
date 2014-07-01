'''
Created on Sep 6, 2013

@author: marco
'''

import csv
import numpy

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
        string = "CsvReader Object:"
        string += "\n-File: "+str(self.filename)
        string += "\n-Columns Available:"
        for c in self.columnNames:
            string += "\n\t-"+str(c)
        string += "\n-Selected: "+str(self.columnSelected)
        return string
      
    def OpenCSV(self, filename=""):
        """
        
        Open a CSV file
        
        """
        # Reinitialize all
        self.__init__(filename)
        
        # Open the file passed as parameter
        try:
            f = open(self.filename, 'rb')
        except IOError:
            print "The file %s does not exist, impossible to open " % self.filename
            return False
        
        # TODO:
        # Read N lines and detect the dialect used
        # N = 1024
        # self.dialect = csv.Sniffer().sniff(self.f.read(N))
        self.dialect.skipinitialspace = True
        
        # Move the file pointer to the beginning
        f.seek(0)
        
        # Real the csv file and instantiate the reader
        try:
            reader = csv.DictReader(f, dialect = self.dialect)
            self.columnNames = reader.fieldnames
            f.close()
            return True
        except csv.Error:
            print "ERROR:: The csv file "+filename+" is not correct, please check it..."
            f.close()
            return False
        
        #TODO: check if the file descriptor has to be closed
    
    def GetFileName(self):
        """
        This method returns the filename of the CSV file associated
        """
        return self.filename
        
    def GetColumnNames(self):
        """
        This method returns a list containing the names of the columns contained in the csv file
        """
        return self.columnNames
    
    def SetSelectedColumn(self, columnName):
        """
        This method allows to specify which is the column to be selected
        """
        if columnName in self.GetColumnNames():
            self.columnSelected = columnName
            return True
        else:
            print "ERROR:: The column selected "+str(columnName)+"is not part of the columns names list"
            print self.GetColumnNames()
            return False
            
    def GetSelectedColumn(self):
        """
        This method returns the column selected for this CSV reader
        """
        if self.columnSelected != None:
            return self.columnSelected
        else:
            return ""
            
    def PrintDialectInformation(self):
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
        
    def GetDataSeries(self):
        """
        Once the csv file and its column have been selected, it is possible to read the data from the csv
        file and return them. Please remember that the first column of the csv file HAS to be the time.
        This method returns a dictionary:
        
        dataSeries = {"time": [0, 1, 2, 3, 4, ...], "data": [12, 34, 33, 12.5, 66, ...]}
         
        """
        dataSeries = {}
        # Check if the file name has been selected
        if self.filename != None and self.filename != "":
            
            # Open the file
            try:
                f = open(self.filename, 'rb')
            except IOError:
                print "Error: The csv file ", self.filename, " cannot be open"
                return dataSeries
            
            # Move the file pointer to the beginning
            f.seek(0)
            
            # Read the csv file and instantiate the reader
            try:
                reader = csv.DictReader(f, dialect = self.dialect)
            except csv.Error:
                print "ERROR:: The csv file ", self.filename, " is not correct, please check it..."
                f.close()
                return dataSeries
            
            # Check if the column name is set
            if self.columnSelected != None:
                
                # Check if the column name is part of the available dictionary
                if self.columnSelected in self.columnNames:
                    
                    # Read the time and data column from the csv file
                    time = []
                    data = []
                    time_key = self.columnNames[0]
                    for line in reader:
                        time.append(float(line[time_key]))
                        data.append(float(line[self.columnSelected]))
                    
                    # If the check of the cSV file is successful, return the data series, otherwise
                    # return an empty dictionary 
                    if self.CheckTimeSeries(time, self.filename):
                        dataSeries[Strings.TIME_STRING] = numpy.array(time).astype(numpy.float)
                        dataSeries[Strings.DATA_STRING] = numpy.matrix(data).astype(numpy.float)
                        
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
    
    @staticmethod
    def CheckTimeSeries(time, fileName):
        """
        This method check if all the time instants of the time series are increasing,
        If there are two or more equal values a report about this error is returned
        """
        N = len(time)
        wrong = False
        message = "\nWRONG CSV FILE: "+str(fileName)
        for i in range(N-1):
            if time[i] >= time[i+1]:
                message += "\n-Row("+str(i+1)+") time step="+str(time[i])
                message += " >= than next row, equal to "+str(time[i+1])
                wrong = True
                
        if wrong:
            print message
            return False
        else:
            return True     