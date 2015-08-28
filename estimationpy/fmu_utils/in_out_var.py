'''
Created on Nov 6, 2013

@author: marco
'''
import numpy
import pandas as pd

from estimationpy.fmu_utils.csv_reader import CsvReader
from estimationpy.fmu_utils import strings
import pyfmi

class InOutVar():
    """
    This class represents a generic input or output variable of a
    dynamic system.
    In case of state and parameter estimation both input and outputs 
    of a system have the characteristic of having measurements associated
    to them. These measurements, together with other informations that can
    quantify their unsertanty, are used by the estimation algorithm.
    
    This class can be seen as a wrapper around the class **pyfmi.fmi.ScalarVariable**
    with the addition of an object that represent a measurement time series.
    The time series can be directly defined as a pandas.Series object or with the
    convenience class :class:`estimationpy.fmu_utils.csv_reader.CsvReader`.
    
    **Note**
    
        When dealing with real systems not all the output measurements can be measured.
        For example a translational mechanic system can have as outputs the position of a mass, 
        it velocity, and its acceleration. However, in a real application one may only have 
        access to one of these measurements.
        The class uses a boolean flag to define whether an output is actually being measured and
        thus can be used by a state estimation algorithm.
    
    """
    
    def __init__(self, object = None):
        """
        Initialization method. This constructor defines the object (PyFmiVariable) that contains the
        information about the variable, the CsvReader class associated to the input (that contains the data), 
        a dictionary called dataSeries = {"time": [], "data": []} that contains the two arrays that represent
        the data series (that are read from the csv file).
        The integer index is used when the data values are read using the function ReadFromDataSeries, while
        cov is the covariance associated to the data series.
        """
        self.object = object
        self.csvReader = CsvReader()
        self.dataSeries = pd.Series()
        
        self.index = 0
        self.cov = 1.0
        self.measOut = False
    
    def read_value_in_fmu(self, fmu):
        """
        Given an FMU model, this method reads the value of the variable/parameter
        """
        type = self.object.type
        if type == pyfmi.fmi.FMI_REAL:
            val = fmu.get_real(self.object.value_reference)
        elif type == pyfmi.fmi.FMI_INTEGER:
            val = fmu.get_integer(self.object.value_reference)
        elif type == pyfmi.fmi.FMI_BOOLEAN:
            val = fmu.get_boolean(self.object.value_reference)
        elif type == pyfmi.fmi.FMI_ENUMERATION:
            val = fmu.get_int(self.object.value_reference)
        elif type == pyfmi.fmi.FMI_STRING:
            val = fmu.get_string(self.object.value_reference)
        else:
            print "OnSelChanged::FMU-EXCEPTION :: The type is not known"
            return None
        return val[0]
    
    def set_measured_output(self, flag = True):
        """
        This method set the flag that indicates if the variable represents a measured output
        """
        self.measOut = flag
    
    def is_measured_output(self):
        """
        This method returns the value of the boolean flag that describe
        if the variable is a measured output
        """
        return self.measOut
    
    def set_covariance(self, cov):
        """
        This method sets the covariance associated to the input/output variable
        """
        if cov > 0.0:
            self.cov = cov
            return True
        else:
            print "The covariance must be positive"
            return False
    
    def get_covariance(self):
        """
        This method returns the covariance of the variable
        """
        return self.cov
     
    def set_object(self, obj):
        """
        Set the object <<pyfmi.fmi.ScalarVariable>> associated to the input/output
        """
        if isinstance(obj, pyfmi.fmi.ScalarVariable):
            self.object = obj
        else:
            raise TypeError("The object passed to the method InOutVar.SetObject() is not of type pyfmi.fmi.ScalarVariable ")
        
    def get_object(self):
        """
        Get the object <<pyfmi.ScalarVariable>> associated to the input/output
        """
        return self.object
    
    def set_csv_reader(self, reader):
        """
        Set the CsvReader class associated to the input/output
        """
        if isinstance(reader, CsvReader):
            self.csvReader = reader
        else:
            msg = "The object passed to the method InOutVar.SetCsvReader() is not of type FmuUtils.CsvReader.CsvReader"
            msg += "\n it is of type %s" % (str(type(reader)))
            raise TypeError(msg)
        
    def get_csv_reader(self):
        """
        Get the CsvReader class associated to the input/output
        """
        return self.csvReader
    
    def read_data_series(self):
        """
        This method reads the data series contained in the CSV file
        """
        # If the CsvReader has been specified the try to load the data from there
        if self.csvReader.filename == "" or self.csvReader.filename == None:
            
            # Check because the dataSeries may have bee specified using a pandas.Series
            if len(self.dataSeries) > 0:
                return True
            else:
                return False
        else:
            
            # Read the data from the CSV
            self.dataSeries = self.csvReader.get_data_series()
            if len(self.dataSeries) > 0:
                return True
            else:
                return False
        
    def get_data_series(self):
        """
        This method returns the data series read from the csv file
        """
        return self.dataSeries
    
    def set_data_series(self, series):
        """
        This function sets a data series instead of reading it from the CSV file.
        The data series has to be formatted as a pandas series that has a 
            pandas.tseries.index.DatetimeIndex
        as index.
        """
        if isinstance(series, pd.Series):
            if isinstance(series.index, pd.DatetimeIndex):
                self.dataSeries = series
            else:
                raise TypeError("The index of the Series passed to the method InOutVar.SetDataSeries() is not of type pandas.DatetimeIndex")
        else:
            raise TypeError("The object passed to the method InOutVar.SetDataSeries() is not of type pandas.Series ")
        
    def read_from_data_series(self, ix, verbose = False):
        """
        This function given a datetime index returns the value of the data series at that point.
        The type of ix is a Timestamp.
        """
        from_start = (ix - self.dataSeries.index[0]).total_seconds()
        to_end = (self.dataSeries.index[-1] - ix).total_seconds()
        
        if from_start < 0.0 or to_end < 0.0:
            # The index ix is not contained in the array, it's either
            # before the start or after the end
            return False
        
        try:
            # try to read directly the index, if it exists it's done
            value = self.dataSeries.loc[ix]
            return value
        
        except KeyError:
            
            # The index ix is not present, an interpolation is needed.
            # Since it is a sequential access, store the last position to reduce the access time for the next iteration
            index = self.index
            N = len(self.dataSeries.index)
            
            # Start the identification of the position of the closest time step
            
            # If len(time) = 10 and index was 2, indexes is [2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
            indexes = numpy.concatenate((numpy.arange(index,N), numpy.arange(index+1)))
            
            # Start the iteration
            if verbose:
                print "\n========="
                print "Indexes = ",indexes
            for i in range(N):
                
                j = indexes[i]
                
                if verbose:
                    print "j=",j
                
                # Get the time values (of type Timestamp)
                T_a = self.dataSeries.index[indexes[i]]
                T_b = self.dataSeries.index[indexes[i+1]]
                
                # Since the array is circular it may be necessary to sweep the values
                # This guarantees that T_0 is always the minimum and T_1 the maximum of 
                # the considered interval
                T_0 = min(T_a, T_b)
                T_1 = max(T_a, T_b)
                
                if verbose:
                    print "Time ",ix," and [",T_0,",",T_1,"]"
                
                # Measure the difference in seconds between the desired index
                # and the two points 
                t_0 = (ix - T_0).total_seconds()
                t_1 = (ix - T_1).total_seconds()
                
                # Skip transition when restarting from the beginning (j == N-1)
                # If T_0 <= ix <= T_1 the exit
                if j != N-1 and t_0 >= 0 and t_1 <= 0:
                    break
                else:
                    # Otherwise go to the next couple of points
                    j += 1
            
            # This takes into account that the array is circular and the algorithm may
            # finish in a zone where the order 
            if j < N-1: 
                index_0 = indexes[i]
                index_1 = indexes[i+1]
            else:
                index_1 = indexes[i]
                index_0 = indexes[i-1]
            
            if verbose:
                print "Picked values are",self.dataSeries.values[index_0]," : ", self.dataSeries.values[index_1]
            
            # Get distances in seconds to compute the linear interpolation
            deltaT = (self.dataSeries.index[index_1] - self.dataSeries.index[index_0]).total_seconds()
            dT0 = (ix - self.dataSeries.index[index_0]).total_seconds()
            dT1 = (self.dataSeries.index[index_1] - ix).total_seconds()
            interpData = (dT0*self.dataSeries.values[index_1] + dT1*self.dataSeries.values[index_0])/deltaT
            
            # Save the index
            self.index = j
            
            return interpData