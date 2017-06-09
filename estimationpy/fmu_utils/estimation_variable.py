'''
@author: Marco Bonvini
'''
import numpy
import pyfmi

import logging
logger = logging.getLogger(__name__)

class EstimationVariable(object):
    '''
    This class represents a variable that is part of an estimation
    problem, and it can be either a state variable or a parameter.
    
    An in stance of class :class:`EstimationVariable` has multiple
    attributes and properties that are used during the estimation process.
    The most relevant attributes are
    
    * ``minValue`` is the minimum value a variable can assume and is
      specified in the model description file of the FMU. For example for a
      temperature measured in Kelvin, the ``minValue`` is 0.0.
      
    * ``maxValue`` is the minimum value a variable can assume and is
      specified in the model description file of the FMU. For example for a
      real number that represents a percentage the ``maxValue`` can be either
      1.0 or 100.0.
      
    * ``cov`` is the covariance :math:`\sigma^2` associated to the variable or
      parameter estimated.
      
    * ``initValue`` is the initial value used at the beginning of the estimation
      algorithm.
      
    * ``constraintLow`` is the lower limit the variable can assume during the estimation
      algorithm. This value is usualy set to impose ad-hoc boundaries to the estimation
      algorithm that may be physically not possible or reasonable.
    
    * ``constraintHigh`` is the upper limit the variable can assume during the estimation
      algorithm. This value is usualy set to impose ad-hoc boundaries to the estimation
      algorithm that may be physically not possible or reasonable.
    
    '''

    def __init__(self, fmi_var, fmu):
        '''
        Constructor of the class. This method takes as arguments
        an **FmiVariable** object and an **FmuModel** object and instantiates
        an :class:`EstimationVariable` object.
        
        :param FmiVariable fmi_var: an object representing a variable of an FMU model
          in PyFMI.
        :param FmuModel fmu: an object representing an FMU model in PyFMI.
        
        :raise TypeError: the method raises a ``TypeError`` if the start value of the
          variable is either missing or equal to ``None``.
        
        @todo: verify if ``value[0]`` can be different from the ``start`` value in a ``FmiVariable``.
        
        '''
        
        # Save data relative to the FMI variable
        self.fmi_var = fmi_var
        self.alias = fmi_var.alias
        self.causality = fmi_var.causality
        self.description = fmi_var.description
        self.name = fmi_var.name
        self.type_var = fmi_var.type
        self.value_reference = fmi_var.value_reference
        self.variability = fmi_var.variability
        
        # Take the start, min and max value of this variable
        # TODO: FMUException is raised due to 'No real value to read'
        logger.debug('Initializing variable: {}'.format(self.name))
        t, value, start, min, max = fmu.get_variable_info_numeric(fmi_var)
        
        # TODO: it can be either value[0] or start. Not yet sure about the difference...
        try:
            if value is not None and value[0]!= start:
                logger.info("Start value is different from value read")
                logger.info("Value read  = {0}".format(value[0]))
                logger.info("Start value = {0}".format(start))
                self.initValue = start
            else:
                self.initValue = start
        except TypeError:
            logger.exception("Missing start value (equal to None)")
            self.initValue = start
        
        # Set the attributes of the object
        self.minValue = min
        self.maxValue = max
        self.cov = 1.0
        self.constraintLow = True
        self.constraintHigh = True
    
    def modify_initial_value_in_fmu(self, fmu):
        """
        This method takes as argument an instance of type **FmuModel**
        and sets the initial value of the variable represented
        by this class to its initial value.
        The method returns True if the value has been set corectly, False
        otherwise.
        
        :param FmuModel fmu: an object representing an FMU model in PyFMI.
        
        :return: The outcome of the set operation, either True or False.
        
        :rtype: bool
        
        """
        t = self.type_var
        try:
            if t == pyfmi.fmi.FMI_REAL:
                fmu.set_real(self.value_reference, self.initValue)
            elif t == pyfmi.fmi.FMI_INTEGER:
                fmu.set_integer(self.value_reference, self.initValue)
            elif t == pyfmi.fmi.FMI_BOOLEAN:
                fmu.set_boolean(self.value_reference, self.initValue)
            elif t == pyfmi.fmi.FMI_ENUMERATION:
                fmu.set_int(self.value_reference, self.initValue)
            elif t == pyfmi.fmi.FMI_STRING:
                fmu.set_string(self.value_reference, self.initValue)
            else:
                logger.error("FMU-EXCEPTION, The FMI variable of type {0} is not known".format(t))
                return False
        except FMUException as e:
            logger.error('FMU-EXCEPTION, Failed to set value of {}'.format(self.name))
            raise e

        return True

    def get_fmi_var(self):
        """
        This method returns the underlying FMI variable object that is managed
        by the instance of this class.
        
        :return: a reference to the underlying FMI variable object
        :rtype: pyfmi.fmi.ScalarVariable
        """
        return self.fmi_var
    
    def read_value_in_fmu(self, fmu):
        """
        This method reads the value of a variable/parameter 
        assumes in a specific FMU object.
        
        :param FMuModel fmu: an object representing an FMU model in PyFMI.
        
        :return: The value of the variable represented by an instance of this class.
          The method returns `None` is the type of the variable is not recognized as one
          of the available ones (Real, Integer, Boolean, Enumeration, String).
        
        :rtype: float, None
        
        """
        t = self.type_var
        if t == pyfmi.fmi.FMI_REAL:
            val = fmu.get_real(self.value_reference)
        elif t == pyfmi.fmi.FMI_INTEGER:
            val = fmu.get_integer(self.value_reference)
        elif t == pyfmi.fmi.FMI_BOOLEAN:
            val = fmu.get_boolean(self.value_reference)
        elif t == pyfmi.fmi.FMI_ENUMERATION:
            val = fmu.get_int(self.value_reference)
        elif t == pyfmi.fmi.FMI_STRING:
            val = fmu.get_string(self.value_reference)
        else:
            msg = "FMU-EXCEPTION, The type {0} is not known".format(t)
            logger.error(msg)
            return None
        return val[0]
    
    def info(self):
        """
        This method return a string that contains a formatted
        representation of the **EstimationVariable** object.
        
        :return: String rerpesentation of the variable being estimated
          and its attributes.
        
        :rtype: string
        
        """
        description = "\nName: "+str(self.name)
        description += "\nV_REF: "+str(self.value_reference)
        description += "\nType: "+str(self.type_var)
        description += "\nInit Value = "+str(self.initValue)
        description += "\nCovariance = "+str(self.cov)
        description += "\nConstraints = ("
        if self.constraintLow:
            description += str(self.minValue)
        else:
            description += "-Inf"
        description += " , "
        if self.constraintHigh:
            description += str(self.maxValue)
        else:
            description += "+Inf"
        description += ")"
        return description
    
    def set_initial_value(self, value):
        """
        This method sets the initial value associated to
        an instance of the class :class:`EstimationVariable` that is used
        by the state and parameter estimation algorithm.
        
        :param float value: The value to be used as initial value in the estimation
          algorithm.
          
        """
        self.initValue = value
    
    def get_initial_value(self):
        """
        This method returns a **numpy.array** containing the initial value of
        the **EstimationVariable** object.
        
        :return: a 1-dimensional numpy array of length equal to 1 containing the
          initial value of the **EstimationVariable**.
          
        :rtype: numpy.Array
        
        """
        return numpy.array(self.initValue)
    
    def set_covariance(self, cov):
        """
        This method sets the covariance associated to
        an instance of the class :class:`EstimationVariable` that is used
        by the state and parameter estimation algorithm.
        
        :param float cov: The value to be used as initial value in the estimation
            algorithm. The value must be positive.
        
        :return: True if the value has been set corectly, False otherwise.
        
        :rtype: bool
        
        """
        if cov > 0.0:
            self.cov = cov
            return True
        else:
            msg = "The covariance must be positive"
            logger.error(msg)
            raise ValueError(msg)
        
    def get_covariance(self):
        """
        This method returns the covariance of the **EstimationVariable** object.
        
        :return: the covariance of the variable.
        :rtype: float
        
        """
        return self.cov
    
    def set_min_value(self, value):
        """
        This method sets the minimum value for the **EstimationVariable** object.
        
        :param float value: The minimum value for the variable.
        """
        self.minValue = value
    
    def set_max_value(self, value):
        """
        This method sets the maximum value for the **EstimationVariable** object.
        
        :param float value: The maximum value for the variable.
        """
        self.maxValue = value
        
    def set_constraint_high(self, value):
        """
        This method sets the upper bound constraint for the **EstimationVariable** object.
        
        :param float value: The upper bound constraint value for the variable.
        """
        self.constraintHigh = value
    
    def set_constraint_low(self, value):
        """
        This method sets the upper bound constraint for the **EstimationVariable** object.
        
        :param float value: The upper bound constraint value for the variable.
        """
        self.constraintLow = value
    
    def get_min_value(self):
        """
        This method returns the minimum value of the **EstimationVariable** object.
        
        :return: the minimum value of the variable.
        :rtype: float
        
        """
        return self.minValue
    
    def get_max_value(self):
        """
        This method returns the maximum value of the **EstimationVariable** object.
        
        :return: the maximum value of the variable.
        :rtype: float
        
        """
        return self.maxValue
    
    def get_constraint_high(self):
        """
        This method returns the upper bound constraint value of the **EstimationVariable** object.
        
        :return: the upper bound constraint of the variable.
        :rtype: float
        
        """
        return self.constraintHigh
    
    def get_constraint_low(self):
        """
        This method returns the lower bound constraint value of the **EstimationVariable** object.
        
        :return: the lower bound constraint of the variable.
        :rtype: float
        
        """
        return self.constraintLow
