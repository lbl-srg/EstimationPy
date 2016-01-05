'''
@author: marco
'''

import pyfmi
import numpy
import pandas as pd
import datetime

from estimationpy.fmu_utils.in_out_var import InOutVar
from estimationpy.fmu_utils.estimation_variable import EstimationVariable

import estimationpy.fmu_utils.strings as fmu_util_strings

import logging
logger = logging.getLogger(__name__)

class Model():
    """
    The class :class:`Model` represents an extension of an FMU model.
    The FMU model is meant to be used for simulation purposes only.
    An object of type **Model** groups multiple information that make
    it useful for a state or parameter estimation algorithm.
    
    The **Model** contains a reference to the particular FMU that is being 
    loaded and will be used to perform simulation, but also contains
    other information such as

    * the parameters of the models to estimate,
    * the state variables of the model to estimate,
    * the inputs and output variables as well the data associated to them.
    
    The parameters and state variables to be stimated will be represented
    by objects of type :class:`estimationpy.fmu_utils.estimation_variable.EstimationVariable`,
    the input and output variables are represented by objects of type
    :class:`estimationpy.fmu_utils.in_out_var.InOutVar`.
    
    """
    
    def __init__(self, fmu_file = None, result_handler = None, solver = None, atol = 1e-6, rtol = 1e-4, verbose = None, offset = None):
        """
        
        Constructor method that initializes an object of type **Model** that can be used for simulation
        purposes or within an estimation algorithm.
        
        :param string fmu_file: String representing the path of an FMU model.
        :param string result_handler: String that specifies how pyfmi should handle the results of 
          the simulations of the FMU model. See :mod:`estimationpy.fmu_utils.strings` for available options.
        :param string solver: String that specifies the solver used by pyfmi to simulate 
          the FMU model. See :mod:`estimationpy.fmu_utils.strings` for available options.
        :param float atol: Absolute tolerance used by the solver when simulating the FMU model.
        :param float rtol: relative tolerance used by the solver when simulating the FMU model.
        :param string verbose: level of verbosity required when log messagess are generated. 
        :param datetime.datetime offset: a datetime object that is used as offset in the simulations. For example
          when running an FMU that references a weather data file, one can specify as offset the midnight of the first
          of January (in ISO format UTC referenced <year>-01-01T00:00+0000) and the command :func:`simulate` will use
          the offset date instead of teh start date to compute the time in seconds.
        """
        
        # Reference to the FMU, that will be loaded using pyfmi
        self.fmu = None
        self.fmu_file = fmu_file
        # List of parameters
        self.parameters = []
        # List of state variables
        self.variables = []
        # List of inputs
        self.inputs = []
        # List of outputs
        self.outputs = []
        
        # Initialize the properties of the FMU
        self.name = ""
        self.author = ""
        self.description = ""
        self.fmu_type = ""
        self.version = ""
        self.guid = ""
        self.tool = ""
        self.numStates = ""
        
        # Number of maximum tries for a simulation to be successfully run
        self.SIMULATION_TRIES = 4
        
        # Empty dictionary that will contain the simulation options
        self.opts = {}
        
        # Set the number of states
        self.N_STATES = 0
        
        # Set the simulation date time offset
        self.offset = offset
        
        # An array that contains the value references for every state variable
        self.stateValueReferences = []
        
        # See what can be done in catching the exception/propagating it
        if fmu_file != None:
            self.__set_fmu__(fmu_file, result_handler, solver, atol, rtol, verbose)
    
    def add_parameter(self, obj):
        """
        This method adds one object to the list of parameters to estimate.
        This list contains only the parameters that will be modified by further
        algorithms that make use of objects of type **Model**.
        
        :param FmiVariable obj: an object representing a variable of the FMU in PyFMI.
          This object, if not already present, is used to instantiate a new object
          of type :class:`estimationpy.fmu_utils.estimation_variable.EstimationVariable`.

        :return: a boolean flag that indicates success. True if added, False if, e.g.,
          the parameter was already present.

        :rtype: bool

        """
        if self.is_parameter_present(obj):
            logger.warn("Parameter: {0} not added, already present".format(obj))
            return False
        else:
            # the object is not yet part of the list, add it            
            par = EstimationVariable(obj, self)
            self.parameters.append(par)
            logger.info("Added parameter: {0}".format(par.get_fmi_var().name))
            logger.debug("(... continue) Added parameter: {0} ({1})".format(obj, par))
            
            return True
    
    def add_variable(self, obj):
        """
        This method adds one object to the list of state variables to estimate.
        This list contains only the states that will be modified by further
        algorithms that make use of objects of type **Model**.
        
        :param FmiVariable obj: an object representing a variable of the FMU in PyFMI.
          This object, if not already present, is used to instantiate a new object
          of type :class:`estimationpy.fmu_utils.estimation_variable.EstimationVariable`.

        :return: a boolean flag that indicates success. True if added, False if, e.g.,
          the variable was already present.

        :rtype: bool

        """
        if self.is_variable_present(obj):
            logger.warn("Variable: {0} not added, already present".format(obj))
            return False
        else:
            # the object is not yet part of the list, add it
            # but before embed it into an EstimationVariable class
            var = EstimationVariable(obj, self)
            self.variables.append(var)
            logger.info("Added variable: {0}".format(var.get_fmi_var().name))
            logger.debug("(... continue) Added variable: {0} ({1})".format(obj, var))
            return True
    
    def check_input_data(self, align = True):
        """
        This method checks if all the data series associated to the inputs
        variables of this model are ready to be used or not.
        In case they're not ready because they are not aligned, this method
        tries to correct them providing an interpolation.
        
        See :func:`check_data_list` for more info.
        
        :param bool align: boolean paramater that indicate whether this method should try to 
          align the input data series in case they aren't.

        :return: True if the data is ready, False otherwise.

        :rtype: bool

        """
        return self.check_data_list(self.inputs, align)
    
    def check_data_list(self, dataList, align = True):
        """
        This method check if all the data series provided by the dataList are ready to be used or not.
        If not because they are not aligned, this method tries to correct them providing an interpolation.
        
        :param List(InOutVar) dataList: a list that contains object of type :class:`InOutVar`.
        :param bool align: boolean parameter that specifies whether the method should simply check is the
          the data is correctly aligned or correct them in case they're not aligned.

        :return: a boolean flag that indicates if the data is valid or not.
        :rtype: bool
        """
                
        # Create a list of data series, one for each input
        dataSeries = []
        for inp in dataList:
            dataSeries.append(inp.get_data_series())
        
        Ninputs = len(dataSeries)
        Tmin = 0.0
        Tmax = 0.0
        Npoints = 0
        match = True
        
        # Scan all the data series
        for i in range(Ninputs):
            
            if i == 0:
                Tmin = dataSeries[i].index[0]
                Tmax = dataSeries[i].index[-1]
            else:
                if not dataSeries[i].index.tolist() == dataSeries[i-1].index.tolist():
                    match = False
                    Tmin = max(Tmin, dataSeries[i].index[0])
                    Tmax = min(Tmax, dataSeries[i].index[-1])
        
        # Check if they match or not           
        if match == False and align:
            
            # At the end of this loop we know
            # which data series has the bigger number of points and will
            # be used as base for the other ones
            MaxPoints = 0
            ind = 0
            for i in range(Ninputs):
                NumP = len(dataSeries[i].ix[0])
                if NumP > MaxPoints:
                    MaxPoints = NumP
                    ind = i
            
            # Select the best index
            new_index = dataSeries[ind].index
            
            # Interpolate using the same datetimeIndex
            for inp in dataList:
                inp.get_data_series().reindex(new_index).interpolate(method='linear')

            logger.error("Problems while matching the data series")
            return False
        else:
            logger.info("Match between data series is OK")
            return True

    def get_constr_obs_states_high(self):
        """
        This method returns a **numpy.array** that contains the upper boundary of the constraints
        over the observed state variables.
        
        :return: An array containing the upper bounds of the constraints for the state variables.
        :rtype: numpy.array

        """
        constrHi = numpy.empty(self.get_num_variables())
        i = 0
        for v in self.variables:
            constrHi[i] = v.get_constraint_high()
            i += 1
        return constrHi
    
    def get_constr_obs_states_low(self):
        """
        This method returns a **numpy.array** that contains the lower boundary of the constraints
        over the observed state variables.
        
        :return: An array containing the lower bounds of the constraints for the state variables.
        :rtype: numpy.array

        """
        constrLow = numpy.empty(self.get_num_variables())
        i = 0
        for v in self.variables:
            constrLow[i] = v.get_constraint_low()
            i += 1
        return constrLow
    
    def get_constr_pars_high(self):
        """
        This method returns a **numpy.array** that contains the upper boundary of the constraints
        over the estimated parameters.
        
        :return: An array containing the upper bounds of the constraints for the estimated parameters.
        :rtype: numpy.array

        """
        constrHi = numpy.empty(self.get_num_parameters())
        i = 0
        for p in self.parameters:
            constrHi[i] = p.get_constraint_high()
            i += 1
        return constrHi
    
    def get_constr_pars_low(self):
        """
        This method returns a **numpy.array** that contains the lower boundary of the constraints
        over the estimated parameters.
        
        :return: An array containing the lower bounds of the constraints for the estimated parameters.
        :rtype: numpy.array

        """
        constrLow = numpy.empty(self.get_num_parameters())
        i = 0
        for p in self.parameters:
            constrLow[i] = p.get_constraint_low()
            i += 1
        return constrLow
    
    def get_cov_matrix_states(self):
        """
        This method returns a **numpy.ndarray** representing the covariance matrix of the estimated
        state variables. The matrix is diagonal.
        
        :return: A matrix that is the covariance matrix of the state variables.
        :rtype: numpy.ndarray

        """
        cov = numpy.diag(numpy.zeros(self.get_num_variables()))
        i = 0
        for v in self.variables:
            cov[i,i] = v.get_covariance()
            i += 1
        return cov
    
    def get_cov_matrix_state_pars(self):
        """
        This method returns a **numpy.ndarray** representing the covariance matrix of the estimated
        state variables and parameters. The matrix is diagonal.
        
        :return: A matrix that is the covariance matrix of the state variables and parameters.
        :rtype: numpy.ndarray

        """
        cov = numpy.diag(numpy.zeros(self.get_num_variables() + self.get_num_parameters()))
        i = 0
        for v in self.variables:
            cov[i,i] = v.get_covariance()
            i += 1
        for p in self.parameters:
            cov[i,i] = p.get_covariance()
            i += 1
        return cov
    
    def get_cov_matrix_parameters(self):
        """
        This method returns a **numpy.ndarray** representing the covariance matrix of the estimated
        parameters. The matrix is diagonal.
        
        :return: A matrix that is the covariance matrix of the estimated parameters.
        :rtype: numpy.ndarray

        """
        cov = numpy.diag(numpy.zeros(self.get_num_parameters()))
        i = 0
        for p in self.parameters:
            cov[i,i] = p.get_covariance()
            i += 1
        return cov
    
    def get_cov_matrix_outputs(self):
        """
        This method returns a **numpy.ndarray** representing the covariance matrix of the measured
        output variables. The matrix is diagonal.
        
        :return: A matrix that is the covariance matrix of the measured output variables.
        :rtype: numpy.ndarray

        """
        cov = numpy.diag(numpy.zeros(self.get_num_measured_outputs()))
        i = 0
        for o in self.outputs:
            if o.is_measured_output():
                cov[i,i] = o.get_covariance()
                i += 1
        return cov
          
    def get_fmu(self):
        """
        This method return the FMU associated to the model.
        
        :return: An object representing the FMU model in PyFMI.
        :rtype: pyfmi.FmuModel
        """
        return self.fmu
    
    def get_fmu_file_path(self):
        """
        This method returns the file path of the FMU.
        
        :return: String representing the path of the FMU file.
        :rtype: string
        """
        return self.fmu_file
    
    def get_fmu_name(self):
        """
        This method returns the name of the FMU associated to the model.
        
        :return: The name of the FMU from its model description XML file.
        :rtype: string
        """
        return self.name
    
    def get_inputs(self):
        """
        Return the list of input variables associated to the FMU that have been selected.
        The variables are of type :class:`estimationpy.fmu_utils.in_out_var.InOutVar`.
        
        :return: List of **InOutVar** objects representing the inputs of the model.
        :rtype: List(estimationpy.fmu_utils.in_out_var.InOutVar)
        
        """
        return self.inputs
    
    def get_input_by_name(self, name):
        """
        This method returns the input variable contained in the list of inputs 
        associated to this model that has a name equal to the parameter ``name``.
        
        :param string name: Name of the input variable to find.

        :return: The input variable associated to the parameter ``name`` if exists, 
          otherwise ``None``.

        :rtype: estimationpy.fmu_utils.in_out_var.InOutVar, None

        """
        for var in self.inputs:
            if var.get_object().name == name:
                return var
        return None
    
    def get_input_names(self):
        """
        This method returns a list containing the names of the input variables associated
        to this model.
        
        :return: list of input names
        :rtype: List(string)
        """
        inputNames = []
        for inVar in self.inputs:
            # inVar is of type InOutVar and the object that it contains is a PyFMI variable
            inputNames.append(inVar.get_object().name)
        return inputNames    
    
    def get_measured_outputs_values(self):
        """
        This method return a vector that contains the values of the observed state variables of the model.
        The observed outputs are listed in the object variable ``self.outputs`` and are identified by the
        flag ``measured = True``.

        :return: A numpy array containing the values of the measured outputs.
        :rtype: numpy.array
        """
        obsOut = numpy.zeros(self.get_num_measured_outputs())
        i = 0
        for o in self.outputs:
            if o.is_measured_output():
                obsOut[i] = o.read_value_in_fmu(self.fmu)
                i += 1
        return obsOut
    
    def get_measured_data_ouputs(self, t):
        """
        The method reads the values of the measured outputs at a given time that is 
        specified by the parameter ``t``.
        The parameter ``t`` is a datetime object that is part of the indexes os the
        pandas.Series associated to the outputs.

        :param datetime.datetime t: time index at which reading the measured output data.

        :return: an array containing the values of the measured outputs at date and time t.
        :rtype: numpy.ndarray
        """
        obsOut = numpy.zeros(shape=(1, self.get_num_measured_outputs()))
        i = 0
        for o in self.outputs:
            if o.is_measured_output():
                obsOut[0,i] = o.read_from_data_series(t)
                i += 1
        return  obsOut
    
    def get_measured_output_data_series(self):
        """
        This method returns a matrix that contains the data values of the measured
        outputs of the model. The matrix contains the data in columns. The first column
        is the time, while the remaining columns are the values of the measured outputs.
        
        :return: a matrix containing the measured outputs and the time at which these values
          are defined.
        :rtype: numpy.ndarray
        """
        # Get all the data series from the CSV files (for every input of the model)
        outDataSeries = []
        for o in self.outputs:
            if o.is_measured_output():
                outDataSeries.append(o)
        
        # Try to align the measured output data
        self.check_data_list(outDataSeries, align = True)
        
        # Now transform it into a matrix with time as first column and the measured outputs
        time = outDataSeries[0].get_data_series().index
        
        # Create the empty matrix
        Npoints = len(time)
        Nouts = self.get_num_measured_outputs()
        dataMatrix = numpy.zeros(shape=(Npoints, Nouts+1))
        
        # Define the first column as the time
        dataMatrix[:,0] = time
        
        # Put the other values in the following columns
        i = 1
        for o in outDataSeries:
            dataMatrix[:,i] = o.get_data_series().values
            i += 1
        
        return dataMatrix 
    
    def get_num_inputs(self):
        """
        This method returns the total number of input variables of the FMU 
        associated to this model.
        
        :return: the number of inputs
        :rtype: int
        """
        return len(self.inputs)
    
    def get_num_outputs(self):
        """
        This method returns the total number of output variables of the FMU 
        associated to this model.
        
        :return: the number of outputs
        :rtype: int
        """
        return len(self.outputs)
    
    def get_num_measured_outputs(self):
        """
        This method returns the total number of measured output variables of the FMU 
        associated to this model.
        
        :return: the number of measured outputs
        :rtype: int
        """
        i = 0
        for o in self.outputs:
            if o.is_measured_output():
                i += 1
        return i
    
    def get_num_parameters(self):
        """
        This method returns the number of parameters of the FMU model that will be 
        estimated or identified.
        
        :return: number of parameters to estimate
        :rtype: int
        """
        return len(self.parameters)
    
    def get_num_variables(self):
        """
        This method returns the number of state variables of the FMU model that will be 
        estimated or identified.
        
        :return: number of states to estimate
        :rtype: int
        """
        return len(self.variables)
    
    def get_num_states(self):
        """
        This method returns the total number of states variables of the FMU 
        associated to this model.
        
        :return: the total number of states
        :rtype: int
        """
        return self.N_STATES
    
    def get_outputs(self):
        """
        Return the list of output variables associated to the FMU.
        
        :return: a list of output variables associated to the model.
        :rtype: List(estimationpy.fmu_utils.in_out_var.InOutVar)
        """
        return self.outputs
    
    def get_output_by_name(self, name):
        """
        This method returns the output variable identified by the
        parameter ``name``. If there is no variable with the same
        name, the method returns None.

        :return: The output variable identified by the parameter ``name``
        :rtype: estimationpy.fmu_utils.in_out_var.InOutVar, None
        """
        for var in self.outputs:
            if var.get_object().name == name:
                return var
        logger.exception("Output variable with name {0} not found".format(name))
        return None
    
    def get_output_names(self):
        """
        This method returns a list of names of the outputs of the model.
        
        :return: a list containing the names of the outputs
        :rtype: List(String)
        """
        outputNames = []
        for outVar in self.outputs:
            # outVar is of type InOutVar and the object that it contains is a PyFMI variable
            outputNames.append(outVar.get_object().name)
        return outputNames
    
    def get_outputs_values(self):
        """
        This method return a vector that contains the values of the outputs as read 
        in the FMU associated to this model. The method uses
        :func:`estimationpy.fmu_utils.in_out_var.InOutVar.read_value_in_fmu`.
        
        :return: an array containing the values of the outputs as read in the FMU model.
        :rtype: numpy.ndarray
        """
        obsOut = numpy.zeros(self.get_num_outputs())
        i = 0
        for o in self.outputs:
            obsOut[i] = o.read_value_in_fmu(self.fmu)
            i += 1
        return obsOut
    
    def get_parameters(self):
        """
        Return the list of parameters of the model that have been selected.
        The selection of a parameter happens with the method
        :func:`add_parameter`.
        
        :return: a list of parameters that are selected
        :rtype: List(estimationpy.fmu_utils.estimation_variable.EstimationVariable)
        """
        return self.parameters
    
    def get_parameters_min(self):
        """
        This method return an array that contains the minimum values that the selected parameters
        can assume.
        
        :return: an array with the min values the parameters can assume.
        :rtype: numpy.array
        """
        minValues = numpy.zeros(self.get_num_parameters())
        i = 0
        for p in self.parameters:
            minValues[i] = p.get_min_value()
            i += 1
        return minValues
    
    def get_parameters_max(self):
        """
        This method return an array that contains the maximum values that the estimated parameters
        can assume.
        
        :return: an array with the max values the parameters can assume.
        :rtype: numpy.array
        """
        maxValues = numpy.zeros(self.get_num_parameters())
        i = 0
        for p in self.parameters:
            maxValues[i] = p.get_max_value()
            i += 1
        return maxValues
    
    def get_parameter_names(self):
        """
        This method returns a list containing the names of the parameters selected
        in this model that will be estimated.
        
        :return: a list containing the names of the selected parameters.
        :rtype: List
        """
        parNames = []
        for par in self.variables:
            # EstimationVariable
            parNames.append(par.name)
        return parNames
    
    def get_parameter_values(self):
        """
        This method return a vector that contains the values of the parametrs as read 
        in the FMU associated to this model. The method uses
        :func:`estimationpy.fmu_utils.estimation_variable.EstimationVariable.read_value_in_fmu`.
        
        :return: an array containing the values of the parameters selected as read in the FMU model.
        :rtype: numpy.ndarray
        """
        obsPars = numpy.zeros(self.get_num_parameters())
        i = 0
        for p in self.parameters:
            obsPars[i] = p.read_value_in_fmu(self.fmu)
            i += 1
        return obsPars
    
    def get_properties(self):
        """
        This method returns a tuple containing the properties of the FMU associated to this model.
        
        :return: A tuple with the following information
        
          - name of the FMU,
          - author of the FMU,
          - description of the FMU,
          - type of the FMU (e.g., Model Exchange),
          - version of the FMU,
          - GUID of the FMU,
          - tool used to generate it,
          - number of states of the FMU

        :rtype: tuple
        """
        return (self.name, self.author, self.description, self.fmu_type, self.version, self.guid, self.tool, self.numStates)
    
    def get_real(self, var):
        """
        Get a real variable specified by the PyFmiVariable from the FMU.
        
        :param pyfmiVariable var: FMI variable for which reading the value.

        :return: the value of the variable
        :rtype: float
        """
        return self.fmu.get_real(var.value_reference)[0]
    
    def get_simulation_options(self):
        """
        This method returns the simulation options of the simulator.
        
        :return: a dictionary that contains the option for the simulator and the solver.
        :rtype: dict
        """
        return self.opts
    
    def get_state(self):
        """
        This method returns an array that contains the values of the entire state variables of the model.
        The methos uses the underlying method ``_get_continuous_states`` provided by 
        PyFMI.
        
        :return: array containing the state vector
        :rtype: numpy.array
        """
        return self.fmu._get_continuous_states()
    
    def get_state_observed_values(self):
        """
        This method returns an array that contains the values of the observed state variables of the model.
        The observed states are listed in ``self.variables``. The order in which they appear is the
        same they were added using the method :func:``add_variable``.
        
        :return: array containing the observed states in the order they were declared.
        :rtype: numpy.array
        """
        obsState = numpy.zeros(self.get_num_variables())
        i = 0
        for v in self.variables:
            obsState[i] = v.read_value_in_fmu(self.fmu)
            i += 1
        return obsState
    
    def get_state_observed_min(self):
        """
        This method return an array that contains the minimum values that the observed states can assume.
        
        :return: an array with the min values the observed states can assume.
        :rtype: numpy.array
        """
        minValues = numpy.zeros(self.get_num_variables())
        i = 0
        for v in self.variables:
            minValues[i] = v.get_min_value()
            i += 1
        return minValues
    
    def get_state_observed_max(self):
        """
        This method return an array that contains the maximum values that the observed states can assume.
        
        :return: an array with the max values the observed states can assume.
        :rtype: numpy.array
        """
        maxValues = numpy.zeros(self.get_num_variables())
        i = 0
        for v in self.variables:
            maxValues[i] = v.get_max_value()
            i += 1
        return maxValues
        
    def get_variables(self):
        """
        Return the list of observed state variables of the FMU associated to the model.
        The selection of an observed state is done by the method :func:`add_variable`.

        :return: list of observed states in the same order they were added with the method
          :func:`add_variable`.
        :rtype: List(estimationpy.fmu_utils.estimation_variable.EstimationVariable)
        """
        return self.variables
    
    def get_variable_info_numeric(self, variable_info):
        """
        This function takes as parameter ``variable_info`` that is an object that 
        can either represent a parameter, a state variable, an input or an output
        and returns the values and the details associated to it in the FMU.
        
        :param estimationpy.fmu_utils.{in_out_var.InOutVar,estimation_variable.EstimationVariable} variable_info: the
          object for which retrieving the data from the FMU associated to the model.
        :return: a tuple with the following values

          1. the Type of the variable,
          2. the value of the variable,
          3. the start value of the variable,
          4. the min value of the variable,
          5. the max value of the variable

          the tuple contains all None values if there's no real value to read for the
          variable.

        :rtype: tuple
        """
        
        try:
            # Take the data type associated to the variable
            t  = self.fmu.get_variable_data_type(variable_info.name)
            
            # According to the data type read, select one of these methods to get the information
            if t == pyfmi.fmi.FMI_REAL:
                value = self.fmu.get_real( variable_info.value_reference )
            elif t == pyfmi.fmi.FMI_INTEGER:
                value = self.fmu.get_integer( variable_info.value_reference )
            elif t == pyfmi.fmi.FMI_BOOLEAN:
                value = self.fmu.get_boolean( variable_info.value_reference )
            elif t == pyfmi.fmi.FMI_ENUMERATION:
                value = self.fmu.get_int( variable_info.value_reference )
            elif t == pyfmi.fmi.FMI_STRING:
                value = self.fmu.get_string( variable_info.value_reference )
            else:
                logger.error("FMU-EXCEPTION, The type {0} is not known".format(t))
                value = 0.0
 
            # TODO: check the min and max value if the variables are not real or integers
            Min   = self.fmu.get_variable_min(variable_info.name)
            Max   = self.fmu.get_variable_max(variable_info.name)
                
            try:
                start = self.fmu.get_variable_start(variable_info.name)
            except pyfmi.fmi.FMUException:
                logger.warn("Default start value defined as 0.0 for variable {0}".format(variable_info.name))
                start = 0.0
            
            return (type, value, start, Min, Max)
        
        except pyfmi.fmi.FMUException:
                # if the real value is not present for this parameter/variable
                logger.error("FMU-EXCEPTION, No real value to read for variable {0}".format(variable_info.name))
                return (None, None, None, None, None)
    
    def get_variable_info(self, variable_info):
        """
        Given a variable_info object that may be related either to a parameter, a state variable, an input or a output
        This function returns the values and details associated to it.
        """
        
        try:
            # Take the data type associated to the variable
            t  = self.fmu.get_variable_data_type(variable_info.name)
            
            # According to the data type read, select one of these methods to get the information
            if t == pyfmi.fmi.FMI_REAL:
                value = self.fmu.get_real( variable_info.value_reference )
                strType = "Real"
            elif t == pyfmi.fmi.FMI_INTEGER:
                value = self.fmu.get_integer( variable_info.value_reference )
                strType = "Integer"
            elif t == pyfmi.fmi.FMI_BOOLEAN:
                value = self.fmu.get_boolean( variable_info.value_reference )
                strType = "Boolean"
            elif t == pyfmi.fmi.FMI_ENUMERATION:
                value = self.fmu.get_int( variable_info.value_reference )
                strType = "Enum"
            elif t == pyfmi.fmi.FMI_STRING:
                value = self.fmu.get_string( variable_info.value_reference )
                strType = "String"
            else:
                logger.error("FMU-EXCEPTION, The type {0} is not known".format(t))
                value = [""]
                strType = "Unknown"
 
            # TODO: check the min and max value if the variables are not real or integers
            Min   = self.fmu.get_variable_min(variable_info.name)
            Max   = self.fmu.get_variable_max(variable_info.name)
                
            try:
                start = str(self.fmu.get_variable_start(variable_info.name))
                fixed = self.fmu.get_variable_fixed(variable_info.name)
                start = start+" (fixed ="+str(fixed)+")"
            except pyfmi.fmi.FMUException:
                start = ""
                
            strVal = str(value[0])
            strMin = str(Min)
            strMax = str(Max)
            if min < -1.0e+20:
                strMin = "-Inf"
            if max > 1.0e+20:
                strMax = "+Inf"
            
            return (strType, strVal, start, strMin, strMax)
        
        except pyfmi.fmi.FMUException:
                # if the real value is not present for this parameter/variable
                logger.error("FMU-EXCEPTION, No real value to read for variable {0}".format(variable_info.name))
                return ("", "", "", "", "")
    
    def get_variable_names(self):
        """
        This method returns a list of names for each state variables observed
        """
        varNames = []
        for var in self.variables:
            # EstimationVariable
            varNames.append(var.name)
        return varNames
    
    def get_variable_object(self, name = None):
        """
        This method returns a PyFMI variable given its name
        """
        if name != None and name != "":
            if self.fmu != None:
                try:
                    return self.fmu.get_model_variables()[name]
                except Exception:
                    logger.error("The variable or parameter: {0} is not available in the list: {1}".format(name, self.fmu.get_model_variables().keys()))
                    return None
            else:
                logger.error("The FMU model has not yet been set. Impossible return the variable {0}".format(name))
                return None
        else:
            logger.error("Impossible to look for the name because it is None or empty")
            return None
    
    def initialize_simulator(self, startTime = None):
        """
        This method performs a simulation of length zero to initialize the model.
        The initialization is needed only once before running the first simulation. Simulations
        after the first do not need to execute the initialization again.
        
        By default the simulation is performed at the initial time of the input data series, but the
        user can specify an other point with the parameter ``startTime``.

        :param datetime.datetime startTime: start time and date where the model should be initialized
        
        :return: a boolean flag that indicates if the initialization was successful
        :rtype: bool
        """
        
        # Load the inputs and check if any problem. If any exits.
        # Align inputs while loading.
        if not self.load_input(align = True):
            return False
        
        # Load the outputs and check if any problems. If any exits.
        if not self.load_outputs():
            return False
        
        # Take the time series: the first because now they are all the same (thanks to alignment)
        time = self.inputs[0].get_data_series().index
        
        # Define the initial time for the initialization
        if startTime == None:
            # Start time not specified, start from the beginning
            index = 0
        else:
            
            # Check that the type of start time is of type datetime
            if not isinstance(startTime, datetime.datetime):
                raise TypeError("The parameter startTime has to be of datetime.datetime type")
                
            # Start time specified, start from the closest point
            if startTime >= time[0] and startTime <= time[-1]:
                index = 0
                for t in time:
                    if t < startTime:
                        index += 1
                    else:
                        break
            else:
                index = 0
                raise IndexError("The value selected as initialization start time is outside the time frame")
                
        # Once the index is know it can be used to define the start_time
        # If the offset is specified then use it as start time
        start_time = time[index]
        
        # Take all the data series
        Ninputs = len(self.inputs)
        start_input = numpy.zeros((1, Ninputs))
        start_input_1 = numpy.zeros((1, Ninputs))
        start_input_2 = numpy.zeros((1, Ninputs))
        i = 0
        if index == 0:
            for inp in self.inputs:
                dataInput = numpy.matrix(inp.get_data_series().values).reshape(-1,1)
                start_input[0, i] = dataInput[index,0]
                i += 1
        else:
            for inp in self.inputs:
                dataInput = numpy.matrix(inp.get_data_series().values).reshape(-1,1)
                start_input_1[0, i] = dataInput[index-1,0]
                start_input_2[0, i] = dataInput[index,0]
                
                # Linear interpolation between the two values
                dt0 = (time[index] - start_time).total_seconds()
                dT1 = (start_time  - time[index-1]).total_seconds()
                DT  = (time[index] - time[index-1]).total_seconds()
                
                # Perform the interpolation
                start_input[0, i] = (dt0*start_input_1[0, i] + dT1*start_input_2[0, i])/DT
                
                i += 1
               
        # Initialize the model for the simulation
        self.opts["initialize"] = True
        
        try:
            # Simulate from the initial time to initial time + epsilon
            # thus we have 2 points
            
            # Create the input objects for the simulation that initializes
            input_u = numpy.hstack((start_input, start_input))
            input_u = input_u.reshape(2,-1)
            
            time = pd.DatetimeIndex([start_time, start_time])
            
            # Run the simulation, remember that
            # time has to be a dateteTimeIndex and Input has to be a numpy.matrix
            self.simulate(time = time, input = input_u)
            self.opts["initialize"] = False
            
            # Initialize the selected variables and parameters to the values indicated 
            # Done after very small simulation because there can be some internal parameters that defines
            # the initial value and may override the initialization with the indicated values
            for v in self.variables:
                v.modify_initial_value_in_fmu(self.fmu)
            for p in self.parameters:
                p.modify_initial_value_in_fmu(self.fmu)
            
            return True
        
        except ValueError:
            logger.error("First simulation for initialize the model failed")
            return False
    
    def is_parameter_present(self, obj):
        """
        This method returns True is the variable specified by the parameter ``obj`` 
        is contained in the list of variables of the model (e.g, observed states).
        
        :return: flag that indicates if the variable is present in the list of variables.
        :rtype: bool
        """
        val_ref = obj.value_reference
        for p in self.parameters:
            if p.value_reference == val_ref:
                # there is already a parameter in the list with the same value_reference
                logger.error("There is already a parameter in the list with the same value reference: {0}".format(val_ref))
                return True
        return False
    
    def is_variable_present(self, obj):
        """
        This method returns True is the variable specified by the parameter ``obj`` 
        is contained in the list of variables of the model (e.g, observed states).
        
        :return: flag that indicates if the variable is present in the list of variables.
        :rtype: bool
        """
        val_ref = obj.value_reference
        for v in self.variables:
            if v.value_reference == val_ref:
                # there is already a variable in the list with the same value_reference
                logger.error("There is already a variable in the list with the same value reference: {0}".format(val_ref))
                return True
        return False
    
    def load_input(self, align = True):
        """
        This method loads all the pandas.Series associated to the inputs.
        The method returns a boolean variable that indicates if the import was successful.
        
        :return: flag that indicates a successful import
        :rtype: bool
        """
        # Get all the data series from the CSV files (for every input of the model)
        LoadedInputs = True
        for inp in self.inputs:
            LoadedInputs = LoadedInputs and inp.read_data_series()
        
        if not LoadedInputs:
            logger.error("An error occurred while loading the inputs")
        else:
            # A check on the input data series should be performed: Are the initial times, final times and number of point
            # aligned? If not perform an interpolation to align them is done. The boolean flag align deals with this.
            logger.info("Inputs loaded, check sanity of the input data series...")
            if not self.check_input_data(align):
                logger.info("Re-Check the input data series...")
                return self.check_input_data(align)
            
        return LoadedInputs
    
    def load_outputs(self):
        """
        This method loads all the pandas.Series associated to the outputs.
        The method returns a boolean variable that indicates if the import was successful.
        
        :return: flag that indicates a successful import
        :rtype: bool
        """

        # Get all the data series from the CSV files (for every input of the model)
        LoadedOutputs = True
        for o in self.outputs:
            if o.is_measured_output():
                LoadedOutputs = LoadedOutputs and o.read_data_series()
                
        if not LoadedOutputs:
            logger.error("An error occurred while loading the outputs")
        else:
            logger.debug("Outputs loaded correctly")
            
        return LoadedOutputs
    
    def re_init(self, fmu_file, result_handler = None, solver = None, atol = 1e-6, rtol = 1e-4, verbose = None):
        """
        This function reinitializes the FMU associated to the model and resets the default
        options for running the simulations.

        :param string fmu_file: path that identifies the FMU to be loaded
        :param string result_handler: type of handler for managing the results
        :param string solver: name of the solver used for the time integration
        :param float atol: absolute tolerance of the solver
        :param float rtol: relative tolerance of the solver
        :param string verbose: verbosity level used by the ODE solver used by PyFMI

        :rtype: None
        """
        logger.info("Previous FMU was: {0}".format(self.fmu))
        logger.info("Reinitialized model with: {0}".format(fmu_file))
        if self.fmu != None:
            self.fmu = None
        self.__init__(fmu_file, result_handler, solver, atol, rtol, verbose)
    
    def remove_parameter(self, obj):
        """
        This method removes one object from the list of parameters.
        This list contains only the parameters that will be modified during 
        further analysis.
        
        :param estimationpy.fmu_utils.estimation_variable.EstimationVariable obj: The parameter to be
          removed
        :return: a boolean that indicates is the operation was successful
        :rtype: bool
        """
        try:
            index = self.parameters.index(obj)
            self.parameters.pop(index)
            return True
        except ValueError:
            # the object cannot be removed because it is not present
            logger.warn("Parameter {0} not present, can't be remove from the list".format(obj))
            return False
        
    def remove_parameters(self):
        """
        This method removes all the objects from the list of parameters.

        :rtype: None
        """
        self.parameters = []
    
    def remove_variable(self, obj):
        """
        This method removes one object from the list of variables.
        This list contains only the state variables that will be modified during 
        further analysis.
        
        :param estimationpy.fmu_utils.estimation_variable.EstimationVariable obj: The variable to be
          removed
        :return: a boolean that indicates is the operation was successful
        :rtype: bool
        """
        try:
            index = self.variables.index(obj)
            self.variables.pop(index)
            return True
        except ValueError:
            # the object cannot be removed because it is not present
            logger.warn("Variable {0} not present, can't be remove from the list".format(obj))
            return False
    
    def remove_variables(self):
        """
        This method removes all the objects from the list of parameters.
        """
        self.variables = []
    
    def unload_fmu(self):
        """
        This method unloads the FMU by deallocating the resources associated to it 
        from the memory.
        This is necessary if an other FMU model needs to be loaded.

        :rtype: None
        """
        del(self.fmu)
    
    def __set_fmu__(self, fmu_file, result_handler, solver, atol, rtol, verbose):
        """
        This method associate an FMU to a model. If the model has already an FMU
        associated to it the method does not complete the assigment.
        The method provides a convenient way to set multiple of the details
        that are needed to run simulation and handle the FMU in the context 
        os estimationpy:

        1. loading the FMU from a file
        2. get the default options for the simulation and modify them if necessary
        3. identify the number os states and their value references
        4. identify properties of the FMU
        5. identify the inputs and outputs of the FMU
        
        See :mod:`estimationpy.fmu_utils.strings` for more information about the
        options.
        
        :param string fmu_file: path that identifies the FMU to be loaded
        :param string result_handler: type of handler for managing the results
        :param string solver: name of the solver used for the time integration
        :param float atol: absolute tolerance of the solver
        :param float rtol: relative tolerance of the solver
        :param string verbose: verbosity level used by the ODE solver used by PyFMI

        :rtype: None
        """
        if self.fmu == None:
            
            #TODO:
            # See what can be done in catching the exception/propagating it
            self.fmu = pyfmi.load_fmu(fmu_file)
                
            # Get the options for the simulation
            self.opts = self.fmu.simulate_options()
            
            # Define the simulation options
            self.set_simulation_options(result_handler, solver, atol, rtol, verbose)
            
            # Define the standard value for the result file
            self.set_result_file(None)
            
            # set the number of states
            self.N_STATES = len(self.get_state())
            
            # get the value references of the state variables
            self.stateValueReferences = self.fmu.get_state_value_references()
            
            # Properties of the FMU
            self.name = str(self.fmu.get_name())
            self.author = str(self.fmu.get_author())
            self.description = str(self.fmu.get_description())
            self.fmu_type = str(self.fmu.__class__.__name__)
            self.version = str(self.fmu.version)
            self.guid = str(self.fmu.get_guid())
            self.tool = str(self.fmu.get_generation_tool())
            [Ncont, Nevt] = self.fmu.get_ode_sizes()
            self.numStates = "( "+str(Ncont)+" , "+str(Nevt)+" )"
            
            # prepare the list of inputs and outputs
            self.__set_inputs__()
            self.__set_outputs__()
            
        else:
            logger.warn("The FMU has already been assigned to this model")
    
    def __set_in_out_var__(self, variability, causality):
        """
        This method identifies a subset of the variables that belong to the FMU depending
        on their charateristics, namely the causality and variability.
        The variables are identified in the following way
        
        +---------------+-----------+-------------+
        | Variable type | causality | variability |
        +===============+===========+=============+
        |    input      |    0      |     None    |
        +---------------+-----------+-------------+
        |    output     |    1      |     None    |
        +---------------+-----------+-------------+
        
        Once the variables are identified, they're added to the list of inputs and outputs
        of this model.

        :param int variability: integer that identifies the variability of a variable in the FMU.
        :param int causality: integer that identifies the causality of a variable in the FMU.

        :rtype: None
        """
        # Take the variable of the FMU that have the specified variability and causality
        # the result is a dictionary which has as key the name of the variable with the dot notation
        # and as element a class of type << pyfmi.fmi.ScalarVariable >>
        # Alias variable removed for clarity.
        dictVariables = self.fmu.get_model_variables(include_alias = False, variability = variability, causality = causality)
            
        for k in dictVariables.keys():
            # The object attached to each leaf of the tree is << dictParameter[k] >>
            # which is of type << pyfmi.fmi.ScalarVariable >>
            
            var = InOutVar()
            var.set_object(dictVariables[k])
            
            if variability == None and causality ==0:
                # input
                self.inputs.append(var)
            if variability == None and causality ==1:
                # output
                self.outputs.append(var)

    def __set_inputs__(self):
        """
        This function scan the FMU and identifies the input variables of the model.
        The method uses :func:`__set_in_out_var__`.
        After this method is called, the model will have a list of :class:`estimationpy.fmu_utils.in_out_var.InOutVar`
        objects associated to its inputs.

        :rtype: None
        """
        self.__set_in_out_var__(None, 0)
        
    def __set_outputs__(self):
        """
        This function scan the FMU and identifies the output variables of the model.
        The method uses :func:`__set_in_out_var__`.
        After this method is called, the model will have a list of :class:`estimationpy.fmu_utils.in_out_var.InOutVar`
        objects associated to its oututs.

        :rtype: None
        """
        self.__set_in_out_var__(None, 1)
    
    def set_result_file(self, file_name):
        """
        This method modifies the name of the file that stores the simulation results.
        
        :param string file_name: name of the file that will store the simulation results.

        :rtype: None
        """
        if file_name!= None:
            self.opts["result_file_name"] = file_name
        else:
            self.opts["result_file_name"] = ""
    
    def set_simulation_options(self, result_handler, solver, atol, rtol, verbose):
        """
        This method sets the options for the simulator used by PyFMI.
        
        :param string result_handler: result handler that manages the data produced by the FMU.
          The optiona can be to store data on files, memory, or in a custom way.
        :param string solver: dolver used to integrate the ODE represented by the FMU
        :param float atol: absolute tolerance of the solver
        :param float rtol: relative tolerance of the numerical solver
        :param string verbose: string that represents the verbosity level used by the numerical solver

        See :mod:`estimationpy.fmu_utils.strings` for more information about the options
        that are available.
        """
        # The result handling can be one of
        # "file", "memory", "custom" (in the latter case a result handler has to be specified)
        # By default they are on memory
        if result_handler != None and result_handler in fmu_util_strings.SIMULATION_OPTION_RESHANDLING_LIST:
            self.opts[fmu_util_strings.SIMULATION_OPTION_RESHANDLING_STRING] = result_handler
        else:
            self.opts[fmu_util_strings.SIMULATION_OPTION_RESHANDLING_STRING] = fmu_util_strings.RESULTS_ON_MEMORY_STRING
        
        
        # Set solver verbose level
        if verbose != None and  verbose in fmu_util_strings.SOLVER_VERBOSITY_LEVELS:
            for s in fmu_util_strings.SOLVER_NAMES_OPTIONS:   
                self.opts[s][fmu_util_strings.SOLVER_OPTION_VERBOSITY_STRING] = verbose
        else:
            for s in fmu_util_strings.SOLVER_NAMES_OPTIONS:   
                self.opts[s][fmu_util_strings.SOLVER_OPTION_VERBOSITY_STRING] = fmu_util_strings.SOLVER_VERBOSITY_QUIET
        
              
        # Set the absolute and relative tolerance of each solver, otherwise the default value
        # is left
        if atol != None and atol > 0 and numpy.isreal(atol):
            for s in fmu_util_strings.SOLVER_NAMES_OPTIONS:   
                self.opts[s][fmu_util_strings.SOLVER_OPTION_ATOL_STRING] = atol
        if rtol != None and rtol > 0 and numpy.isreal(rtol):
            for s in fmu_util_strings.SOLVER_NAMES_OPTIONS:   
                self.opts[s][fmu_util_strings.SOLVER_OPTION_RTOL_STRING] = rtol
        
    def set_state(self, states_vector):
        """
        This method sets the value of a real variable in the FMU associated to the model.
        
        :param numpy.array states_vector: a numpy array containing the values for all the
          state variables of the FMU associated to the model.

        :rtype: None
        """
        self.fmu._set_continuous_states(states_vector)
    
    def set_real(self, var, value):
        """
        This method sets the value of a real variable in the FMU associated to the model.
        
        :param pyfmi.FMIVariable var: FMi variable that is part of the FMU for which the
          value has to be set
        :param float value: real value to assign to the variable ``var`` in the FMU.
        
        :rtype: None
        """
        self.fmu.set_real(var.value_reference, value)
        
    
    def set_state_selected(self, v):
        """
        This method sets the values of the states to be estimated that are part
        of the list ``self.variables``.

        :param list, np.array s: iterable and oredered sequence of state values to assign
        :return: flag that indicates the success of the operation
        :rtype: bool
        """
        if len(v) == len(self.variables):
            # The vector have compatible dimensions
            i = 0
            for var in self.variables:
                self.fmu.set_real(var.value_reference, v[i])
                i += 1
            return True
        else:
            # the vectors are not compatibles
            logger.error("The vector containing the states to set is not compatible with the number of states selected")
            logger.error("{0} vs {1}".format(len(v), len(self.variables)))
            return False
    
    def set_parameters_selected(self, p):
        """
        This method sets the values of the parameters to be estimated that are part
        of the list ``self.parameters``.

        :param list, np.array p: iterable and oredered sequence of parameter values to assign
        :return: flag that indicates the success of the operation
        :rtype: bool
        """
        if len(p) == len(self.parameters):
            # The vector have compatible dimensions
            i = 0
            for par in self.parameters:
                self.fmu.set_real(par.value_reference, p[i])
                i += 1
            return True
        else:
            # the vectors are not compatibles
            logger.error("The vector containing the parameters to set is not compatible with the number of parameters selected")
            logger.error("{0} vs {1}".format(len(p), len(self.parameters)))
            return False
    
    def simulate(self, start_time = None, final_time = None, time = pd.DatetimeIndex([]), input = None, complete_res = False):
        """
        This method simulates the model from the start time to the final time. The simulation is handled
        by PyFMI and its options can be specified with :func:`set_simulation_options`.
        The method performs the following steps:

        1. Loads all the pandas.Series associated to the inputs variables,
        2. reads all the values associated to the inputs within the period\
          specified by the start and final times,
        3. converts datetime objects into seconds,
        4. creates the input data matrix required by PyFMI,
        5. loads the simulation options,
        6. runs the simulation,
        7. process and aggregates the results

        The inputs can be either read from the pandas.Series associated to the InOutVar objects
        associated to the inputs of the model, or directly specified by the parameter ``input``.
        In such a case, the input data matrix follows the following structure::
        
            input = [[u1(T0), u2(T0), ...,uM(T0)],
                     [u1(T1), u2(T1), ...,uM(T1)],
                     ...
                     [u1(Tend), u2(Tend), ...,uM(Tend)]]
        
        To summarize one can either implicitly use the data associated to the input variables
        and thus specify the parameters ``start_time`` and ``final_time``, otherwise one can
        specify the parameter ``time`` and the input matrix ``input``.
        If none of them are specified the method reads the time index associated to the 
        input variables and identifies the longest period over which all input variables
        are defined.
        
        **NOTE**
        Since it may happen that a simulation fails without apparent reasons, it is better to 
        simulate the model again before actual an error. The number of tries is specified by the
        propertiy ``self.SIMULATION_TRIES``.
        
        :param datetime.datetime start_time: start date and time of the simulation period.
        :param datetime.datetime final_time: end date and time of the simulation period.
        :param pandas.DatetimeIndex time: date and time index that can be used in conjunction with the 
          parameter ``input``.
        :param numpy.ndarray input: matrix that contains the inputs for the model stacked in columns.
        :param bool complete_res: this flag indicates in which format the results should be arranged.
          
          - If ``complete_res == True``, then the method returns all the results as provided by PyFMI.
          - Otherwise it only returns the data that belong to the following categories: state variables,\
            observed state variables, estimated parameters, measured outputs, and outputs.
        
        :return: returns a tuple containing as first element the time instants where the solution of the 
          differential equations are computed by the time integrator. The elements of the time
          vector are datetime objects. The second element is a dictionary containing the results.
          The number of results available depends depends on the value of the parameter ``complete_res``.
        :rtype: tuple
        """
        
        # Number of input variables needed by the model
        Ninputs = len(self.inputs)
        
        # Check if the parameter time has been provided
        if len(time) == 0:
            # Take the time series: the first because now they are all the same
            time = self.inputs[0].get_data_series().index
        else:
            # Check that the type of the time vector is of type pd.DatetimeIndex
            if not isinstance(time, pd.DatetimeIndex):
                raise TypeError("The parameter time has to be a vector of type pd.DatetimeIndex")
            
        # Define initial start time in seconds
        if start_time == None:
            start_time = time[0]
        else:
            # Check that the type of start time is of type datetime
            if not isinstance(start_time, datetime.datetime):
                raise TypeError("The parameter start_time is of type %s, it has to be of datetime.datetime type." % (str(start_time)))
            # Check if the start time is within the range
            if not (start_time >= time[0] and start_time <= time[-1]):
                raise IndexError("The value selected as initialization start time is outside the time frame")
        
        # If the offset is defined, the start time in seconds needs to reference
        # the offset instead of the first time stamp
        if self.offset:
            start_time_sec = (start_time - self.offset).total_seconds()
        else:
            start_time_sec = (start_time - time[0]).total_seconds()
        
        # Define the final time in seconds
        if final_time == None:
            final_time = time[-1]
        else:
            # Check that the type of start time is of type datetime
            if not isinstance(final_time, datetime.datetime):
                raise TypeError("The parameter final_time is of type %s, it has to be of datetime.datetime type." % (str(start_time)))
            # Check if the final time is within the range
            if not (final_time >= time[0] and final_time <= time[-1]):
                raise IndexError("The value selected as initialization start time is outside the time frame")
            # Check that the final time is after the start time
            if not (final_time >= start_time):
                raise IndexError("The final_time %s has to be after the start time %s." % \
                                 (str(final_time), str(start_time)))
                
        # If the offset is defined, the final time in seconds needs to reference
        # the offset instead of the first time stamp
        if self.offset:
            final_time_sec = (final_time - self.offset).total_seconds()
        else:
            final_time_sec = (final_time - time[0]).total_seconds()
        
            
        # Transforms to seconds with respect to the first element, again
        # if the offset is defined it needs to be used as reference
        Npoints = len(time)
        time_sec = numpy.zeros((Npoints,1))
        for i in range(Npoints):
            if self.offset:
                time_sec[i,0] = (time[i] - self.offset).total_seconds()
            else:
                time_sec[i,0] = (time[i] - time[0]).total_seconds()
        
        # Convert to numpy matrix in case it will be stacked in a matrix
        time_sec = numpy.matrix(time_sec)
        
        # Reshape to be consistent
        time_sec  = time_sec.reshape(-1, 1)
        
        if input == None:
            # Take all the data series
            inputMatrix = numpy.matrix(numpy.zeros((Npoints, Ninputs)))
            
            i = 0
            for inp in self.inputs:
                dataInput = numpy.matrix(inp.get_data_series().values).reshape(-1,1)
                inputMatrix[:, i] = dataInput[:,:]
                i += 1
            # Define the input trajectory
            V = numpy.hstack((time_sec, inputMatrix))
            
        else:
            # Reshape to be consistent
            input = input.reshape(-1, Ninputs)
            # Define the input trajectory
            V = numpy.hstack((time_sec, input))
        
        # The input trajectory must be an array, otherwise pyfmi does not work
        u_traj  = numpy.array(V)
        
        # Create input object
        names = self.get_input_names()
        input_object = (names, u_traj)
        
        # Start the simulation
        simulated = False
        i = 0
        while not simulated and i < self.SIMULATION_TRIES:
            try:
                res = self.fmu.simulate(start_time = start_time_sec, input = input_object, final_time = final_time_sec, options = self.opts)
                simulated = True
            except ValueError:
                logger.debug("Simulation of the model from {0} to {1} failed, try again".format(start_time_sec, final_time_sec))
                i += 1
            except Exception, e:
                logger.warn("Exception during simulation: {0}".format(str(e)))
                logger.warn("Simulation of the model failed between {0} and {1}, try again".format(start_time, final_time))
                i += 1 
        
        # Check if the simulation has been done, if not through an exception
        if not simulated:
            logger.error("Not possible to simulate the model, more than {0} unsuccessful tries".format(self.SIMULATION_TRIES))
            logger.error("Error log from PyFMI: {0}".format(self.fmu.get_log()))
            raise Exception
        
        # Obtain the results
        # TIME in seconds has to be converted to datetime
        # and it has to maintain the same offset specified by the input time series in t[0]
        if self.offset:
            offset_res = self.offset - pd.to_datetime(0, unit = 's', utc = True)
            t = pd.to_datetime(res[fmu_util_strings.TIME_STRING], unit="s", utc = True) + offset_res
        else:
            offset_res = time[0] - pd.to_datetime(res[fmu_util_strings.TIME_STRING][0], utc = True)
            t = pd.to_datetime(res[fmu_util_strings.TIME_STRING], unit="s", utc = True) + offset_res
        
        # Get the results, either all or just the selected ones
        if complete_res == False:
            # OUTPUTS
            output_names = self.get_output_names()
            results = {}
            for name in output_names:
                results[name] = res[name]
            # STATES OBSERVED
            var_names = self.get_variable_names()
            for name in var_names:
                results[name] = res[name]
            # PARAMETERS
            par_names = self.get_parameter_names()
            for name in par_names:
                results[name] = res[name]
            
            # THE OVERALL STATE
            results["__ALL_STATE__"]=self.get_state()
            results["__OBS_STATE__"]=self.get_state_observed_values()
            results["__PARAMS__"]=self.get_parameter_values()
            results["__OUTPUTS__"]=self.get_measured_outputs_values()
            results["__ALL_OUTPUTS__"]=self.get_outputs_values()
            
        else:
            # All the results are given back
            results = res
            
        # Return the results
        return (t, results)
    
    def __str__(self):
        """
        This method overrides the default __str__ method for this class
        and creates a string representation for it.
        
        :return: string representation of the object that contains the following information
        
          - FMU file path,
          - name of the FMU,
          - Author of the FMU,
          - Description of the FMU,
          - Type of the FMU,
          - Version of the FMU,
          - GUID of the FMU,
          - Tool used to generate the FMU,
          - Total number of state variables

        :rtype: string
        """
        string = "\nFMU based Model:"
        string += "\n-File: "+str(self.fmu_file)
        string += "\n-Name: "+self.name
        string += "\n-Author: "+self.author
        string += "\n-Description: "+ self.description
        string += "\n-Type: "+self.fmu_type
        string += "\n-Version: "+self.version
        string += "\n-GUID: "+self.guid
        string += "\n-Tool: "+self.tool
        string += "\n-NumStates: "+self.numStates+"\n"
        return string
    
    def toggle_parameter(self, obj):
        """
        This method checks if a parameter
        has been added to the list of parameters to estimate.
        In case it's part of the list, the parameter is removed, otherwise it is
        added.
        
        **NOTE** This method can be useful when coupled with a GUI.

        :rtype: None
        """
        if self.is_parameter_present(obj):
            self.remove_parameter(obj)
        else:
            self.add_parameter(obj)
    
    def toggle_variable(self, obj):
        """
        This method checks if a state variable
        has been added to the list of variable to estimate (i.e., an observed state).
        In case it's part of the list, the variable is removed, otherwise it is
        added.

        **NOTE** This method can be useful when coupled with a GUI.

        :rtype: None
        """
        if self.is_variable_present(obj):
            self.remove_variable(obj)
        else:
            self.add_variable(obj)
