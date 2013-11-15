'''
Created on Nov 11, 2013

@author: marco
'''
import Model
import numpy
import pyfmi

class EstimationVariable():
    '''
    classdocs
    '''

    def __init__(self, object, fmu):
        '''
        Constructor
        '''
        self.object = object
        self.alias = object.alias
        self.causality = object.causality
        self.description = object.description
        self.name = object.name
        self.type = object.type
        self.value_reference = object.value_reference
        self.variability = object.variability
        
        # Take the start, min and max value of this variable
        type, value, start, min, max = fmu.GetVariableInfo_Numeric(object)
        
        # TODO: it can be either value[0] or start. Not yet sube about the difference...
        if value[0]!= start:
            print "Start value is different from value read"
            print "Value read  =",value[0]
            print "Start value =",start
            self.initValue = start
        else:
            self.initValue = start
        self.minValue = min
        self.maxValue = max
        self.cov = 1.0
        self.constraintLow = True
        self.constraintHigh = True
    
    def ModifyInitialValueInFMU(self, fmu):
        """
        Given an FMU model, this method sets the value of the variable/parameter to
        the one indicated by the initialValue
        """
        type = self.type
        if type == pyfmi.fmi.FMI_REAL:
            fmu.set_real(self.value_reference, self.initValue)
        elif type == pyfmi.fmi.FMI_INTEGER:
            fmu.set_integer(self.value_reference, self.initValue)
        elif type == pyfmi.fmi.FMI_BOOLEAN:
            fmu.set_boolean(self.value_reference, self.initValue)
        elif type == pyfmi.fmi.FMI_ENUMERATION:
            fmu.set_int(self.value_reference, self.initValue)
        elif type == pyfmi.fmi.FMI_STRING:
            fmu.set_string(self.value_reference, self.initValue)
        else:
            print "OnSelChanged::FMU-EXCEPTION :: The type is not known"
            return False
        return True
    
    def ReadValueInFMU(self, fmu):
        """
        Given an FMU model, this method reads the value of the variable/parameter
        """
        type = self.type
        if type == pyfmi.fmi.FMI_REAL:
            val = fmu.get_real(self.value_reference)
        elif type == pyfmi.fmi.FMI_INTEGER:
            val = fmu.get_integer(self.value_reference)
        elif type == pyfmi.fmi.FMI_BOOLEAN:
            val = fmu.get_boolean(self.value_reference)
        elif type == pyfmi.fmi.FMI_ENUMERATION:
            val = fmu.get_int(self.value_reference)
        elif type == pyfmi.fmi.FMI_STRING:
            val = fmu.get_string(self.value_reference)
        else:
            print "OnSelChanged::FMU-EXCEPTION :: The type is not known"
            return None
        return val[0]
    
    def Info(self):
        """
        Method that return a string representation of the estimation variable
        """
        description = "\nName: "+str(self.name)
        description += "\nV_REF: "+str(self.value_reference)
        description += "\nType: "+str(self.type)
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
    
    def SetInitialValue(self, value):
        """
        This method sets the initial value of the estimation variable
        """
        self.initValue = value
        
    def SetCovariance(self, cov):
        """
        This method sets the covariance associated to the estimation variable
        """
        if cov > 0.0:
            self.cov = cov
            return True
        else:
            print "The covariance must be positive"
            self.cov = cov
            return False
    
    def SetMinValue(self, value):
        self.minValue = value
    
    def SetMaxValue(self, value):
        self.maxValue = value
        
    def SetConstraintHigh(self, setConstr):
        self.constraintHigh = setConstr
    
    def SetConstraintLow(self, setConstr):
        self.constraintLow = setConstr
    
    def GetInitialValue(self):
        return numpy.array(self.initValue)
    
    def GetMinValue(self):
        return self.minValue
    
    def GetMaxValue(self):
        return self.maxValue
    
    def GetConstraintHigh(self):
        return self.constraintHigh
    
    def GetConstraintLow(self):
        return self.constraintLow

if __name__=="__main__":
    
    # Assign an existing FMU to the model
    filePath = "../../modelica/FmuExamples/Resources/FMUs/FirstOrder.fmu"
    
    # Initialize the FMU model empty
    m = Model.Model(filePath)
    
    # get a variable
    obj = m.GetVariableObject("x")
    
    print obj
    print obj.name
    print obj.value_reference
    
    # new estimation variable
    var = EstimationVariable(obj)
    
    print var
    print var.name
    print var.value_reference