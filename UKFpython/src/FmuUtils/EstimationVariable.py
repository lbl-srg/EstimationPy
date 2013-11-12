'''
Created on Nov 11, 2013

@author: marco
'''
import Model

class EstimationVariable():
    '''
    classdocs
    '''


    def __init__(self, object):
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
        
        self.initValue = 0.0
        self.cov = 1.0
        self.minValue = 0.0
        self.maxValue = 0.0
        self.constraintLow = False
        self.constraintHigh = False
    
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
        return self.initValue
    
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