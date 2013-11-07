'''
Created on Sep 6, 2013

@author: marco
'''

import pyfmi
import numpy
from FmuUtils.InOutVar import InOutVar
from FmuUtils.Tree import Tree
from FmuUtils import Strings

class Model():
    """
    
    This class contains a reference to a particular FMU that has been loaded into the tool.
    For this FMU, several information are collected together. These information describe what to do with the FMU in 
    the future steps.
    
    A list of:
    
    - parameters,
    - and state variables
    
    that will be estimated/identified.
    
    A list of:
    
    - inputs data series,
    - and output data series
    
    that will be used for simulating the FMU and comparing results.
    
    """
    
    def __init__(self, fmuFile = None):
        """
        
        Constructor of the class Model.
        
        """
        
        # Reference to the FMU, that will be loaded using pyfmi
        self.fmu = None
        self.fmuFile = fmuFile
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
        self.type = ""
        self.version = ""
        self.guid = ""
        self.tool = ""
        self.numStates = ""
        
        # Trees that describe the parameter, state variables, input and output hierarchy
        self.treeParameters = Tree(Strings.PARAMETER_STRING)
        self.treeVariables = Tree(Strings.VARIABLE_STRING)
        self.treeInputs = Tree(Strings.INPUT_STRING)
        self.treeOutputs = Tree(Strings.OUTPUT_STRING)
        
        # Number of maximum tries for a simulation to be successfully run
        self.SIMULATION_TRIES = 4
        # Empty dictionary that will contain the simulation options
        self.opts = {}
        
        # Set the number of states
        self.N_STATES = 0
        
        # See what can be done in catching the exception/propagating it
        if fmuFile != None:
            self.__SetFMU(fmuFile)
            self.__SetTrees()
    
    def GetFmuFilePath(self):
        """
        This method returns the filepath of the FMU
        """
        return self.fmuFile
    
    def SetResultFile(self, fileName):
        """
        This method modifies the name of the file that stores the simulation results
        """
        if fileName!= None:
            self.opts["result_file_name"] = fileName
        else:
            self.opts["result_file_name"] = ""
    
    def __SetFMU(self, fmuFile):
        """
        This method associate an FMU to a model, if not yet assigned
        """
        if self.fmu == None:
            #TODO:
            # See what can be done in catching the exception/propagating it
            self.fmu = pyfmi.load_fmu(fmuFile)
            
            # Get the options for the simulation
            self.opts = self.fmu.simulate_options()
            
            # Define the standard value for the result file
            self.SetResultFile(None)
            
            # set the number of states
            self.N_STATES = len(self.GetState())
            
            # Properties of the FMU
            self.name = str(self.fmu.get_name())
            self.author = str(self.fmu.get_author())
            self.description = str(self.fmu.get_description())
            self.type = str(self.fmu.__class__.__name__)
            self.version = str(self.fmu.version)
            self.guid = str(self.fmu.get_guid())
            self.tool = str(self.fmu.get_generation_tool())
            [Ncont, Nevt] = self.fmu.get_ode_sizes()
            self.numStates = "( "+str(Ncont)+" , "+str(Nevt)+" )"
            
            # prepare the list of inputs and outputs
            self.__SetInputs()
            self.__SetOutputs()
            
        else:
            print "WARNING: The fmu has already been assigned to this model! Check your code!"
    
    def ReInit(self, fmuFile):
        """
        This function reinitializes the FMU associated to the model
        """
        print "Previous FMU was: ",self.fmu
        print "Reinitialized model with: ",fmuFile
        if self.fmu != None:
            self.fmu = None
        self.__init__(fmuFile)
    
    def GetState(self):
        """
        This method return a vector that contains the values of the entire state variables of the model
        """
        return self.fmu._get_continuous_states()
    
    def SetState(self, stateVector):
        """
        This method sets the entire state variables vector of the model
        """
        self.fmu._set_continuous_states(stateVector)
    
    def GetInputNames(self):
        """
        This method returns a list of names for each input
        """
        inputNames = []
        for inVar in self.inputs:
            # inVar is of type InOutVar and the object that it contains is a PyFMI variable
            inputNames.append(inVar.GetObject().name)
        return inputNames
    
    def GetOutputNames(self):
        """
        This method returns a list of names for each output
        """
        outputNames = []
        for outVar in self.outputs:
            # outVar is of type InOutVar and the object that it contains is a PyFMI variable
            outputNames.append(outVar.GetObject().name)
        return outputNames
    
    def InitializeSimulator(self, start_time, start_input):
        """
        This method performs a very short simulation to initialize the model
        The next times the model will be simulated without the initialization phase
        """
        self.opts["initialize"] = True
        try:
            input = numpy.hstack((start_input, start_input))
            self.Simulate([start_time, start_input+1e-10], input)
            self.opts["initialize"] = False
            return True
        except ValueError:
            print "First simulation for initialize the model failed"
            return False
    
    def Simulate(self, time, input):
        """
        This method simulates the model from the start_time to the final_time, using a given set of simulation
        options. Since it may happen that a simulation fails without apparent reason (!!), it is better to 
        simulate again the model if an error occurs. After N_TRIES it stops.
        input = [[u1(T0), u2(T0), ...,uM(T0)],
                 [u1(T1), u2(T1), ...,uM(T1)],
                 ...
                 [u1(Tend), u2(Tend), ...,uM(Tend)]]
        """
        
        # Define the input trajectory
        row, col = numpy.size(input)
        V = time
        for j in range(col):
            V = numpy.vstack((V, input[:,j]))
        u_traj  = numpy.transpose(V)
        
        # Create input object
        names = self.GetInputNames()
        input_object = (names, u_traj)
        
        # assign the initial inputs
        i = 0
        for name in names:
            self.fmu.set(name,input[0,i])
            i += 1
        
        # start the simulation
        simulated = False
        i = 0
        while not simulated and i < self.SIMULATION_TRIES:
            try:
                res = self.model.simulate(start_time = time[0], input = input_object, final_time = time[-1], options = self.opts)
                simulated = True
            except ValueError:
                print "Simulation of the model failed, try again"
                i += 1
        
        # Check if the simulation has been done, if not through an exception
        if not simulated:
            raise Exception
        
        # Obtain the results
        t     = res['time']
        output_names = self.GetOutputNames()
        results = {}
        for name in output_names:
            results[name] = res[name]
        
        # Return the results
        return (t, results)
    
    def GetProperties(self):
        """
        This method returns a tuple containing the properties of the FMU
        """
        return (self.name, self.author, self.description, self.type, self.version, self.guid, self.tool, self.numStates)
    
    def GetFmuName(self):
        """
        This method returns the name of the FMU associated to the model
        """
        return self.name
    
    def GetFMU(self):
        """
        This method return the FMU associated to the model
        """
        return self.fmu
       
    def GetParameters(self):
        """
        Return the list of parameters contained by the FMU that have been selected
        """
        return self.parameters
    
    def GetVariables(self):
        """
        Return the list of state variables contained by the FMU that have been selected
        """
        return self.variables
    
    def GetInputs(self):
        """
        Return the list of input variables associated to the FMU that have been selected
        """
        return self.inputs
    
    def GetOutputs(self):
        """
        Return the list of output variables associated to the FMU that have been selected
        """
        return self.outputs
    
    def GetVariableInfo(self, variableInfo):
        """
        Given a variableInfo object that may be related either to a parameter, a state variable, an input or a output
        This function returns the values and details associated to it.
        """
        
        try:
            # Take the data type associated to the variable
            type  = self.fmu.get_variable_data_type(variableInfo.name)
            
            # According to the data type read, select one of these methods to get the information
            if type == pyfmi.fmi.FMI_REAL:
                value = self.fmu.get_real( variableInfo.value_reference )
                strType = "Real"
            elif type == pyfmi.fmi.FMI_INTEGER:
                value = self.fmu.get_integer( variableInfo.value_reference )
                strType = "Integer"
            elif type == pyfmi.fmi.FMI_BOOLEAN:
                value = self.fmu.get_boolean( variableInfo.value_reference )
                strType = "Boolean"
            elif type == pyfmi.fmi.FMI_ENUMERATION:
                value = self.fmu.get_int( variableInfo.value_reference )
                strType = "Enum"
            elif type == pyfmi.fmi.FMI_STRING:
                value = self.fmu.get_string( variableInfo.value_reference )
                strType = "String"
            else:
                print "OnSelChanged::FMU-EXCEPTION :: The type is not known"
                value = [""]
                strType = "Unknown"
 
            # TODO: check the min and max value if the variables are not real or integers
            min   = self.fmu.get_variable_min(variableInfo.name)
            max   = self.fmu.get_variable_max(variableInfo.name)
                
            try:
                start = str(self.fmu.get_variable_start(variableInfo.name))
                fixed = self.fmu.get_variable_fixed(variableInfo.name)
                start = start+" (fixed ="+str(fixed)+")"
            except pyfmi.fmi.FMUException:
                start = ""
                
            strVal = str(value[0])
            strMin = str(min)
            strMax = str(max)
            if min < -1.0e+20:
                strMin = "-Inf"
            if max > 1.0e+20:
                strMax = "+Inf"
            
            return (strType, strVal, start, strMin, strMax)
        
        except pyfmi.fmi.FMUException:
                # if the real value is not present for this parameter/variable
                print "OnSelChanged::FMU-EXCEPTION :: No real value to read for this variable"
                return ("", "", "", "", "")
    
    def GetTree(self, objectTree, variability, causality, onlyStates = False, pedantic = False):
        """
        This function, provided one tree, populates it.
        The tree is used to represent the parameters, variables, input, outputs with the dot notation,
        and used as support for the graphical object tree
        """
        try:
            
            # Take the variable of the FMU that have the specified variability and causality
            # the result is a dictionary which has as key the name of the variable with the dot notation
            # and as element a class of type << pyfmi.fmi.ScalarVariable >>
            # Alias variable removed for clarity.
            dictParameter = self.fmu.get_model_variables(include_alias = False, variability = variability, causality = causality)
            
            if onlyStates:
                # This method return a list containing the reference value of the state variables
                states = self.fmu.get_state_value_references()
                if pedantic:
                    print "Ref. values of the states: "+str(states)
            
            for k in dictParameter.keys():
                ####################################################################################
                # TODO: check if it's true to don't add now the variables which have derivatives
                ####################################################################################
                if "der(" not in k:
                    
                    # Split the variable name that is written with the dot notation
                    strNames = k.split(".")
                    
                    # Given the vector of names obtained with the dot notation creates the branches of the tree
                    # and name correctly each node and leaf.
                    #
                    # The object attached to each leaf of the tree is << dictParameter[k] >>
                    # which is of type << pyfmi.fmi.ScalarVariable >>
                    if onlyStates:
                        
                        # Add the variables that are in the state vector of the system
                        if dictParameter[k].value_reference in states:
                            objectTree.addFromString(strNames, dictParameter[k])
                            if pedantic:
                                print str(k) + " with Ref. value =" + str(dictParameter[k].value_reference)
                                print str(k) + " with Name =" + str(dictParameter[k].name)
                                print dictParameter[k]
                            
                    else:
                        objectTree.addFromString(strNames, dictParameter[k])
            
            if pedantic:
                print objectTree.getAll()
            
            return True
        
        except IndexError:
            # An error can occur if the FMU has not yet been loaded
            print "FMU not yet loaded..."
            return False
    
    def GetTreeByType(self, type):
        """
        This method given a string that describes a given type of tree, it returns that tree
        """
        if type == Strings.PARAMETER_STRING:
            return self.treeParameters
        if type == Strings.VARIABLE_STRING:
            return self.treeVariables
        if type == Strings.INPUT_STRING:
            return self.treeInputs
        if type == Strings.OUTPUT_STRING:
            return self.treeOutputs
        else:
            print "No Match between the type passes and the available trees"
            return None
         
    def GetParametersTree(self):
        """
        This method returns the tree associated to the parameters of the model
        """
        return self.treeParameters
    
    def GetVariablesTree(self):
        """
        This method returns the tree associated to the state variables of the model
        """
        return self.treeVariables
    
    def GetInputsTree(self):
        """
        This method returns the tree associated to the inputs of the model
        """
        return self.treeInputs
    
    def GetOutputsTree(self):
        """
        This method returns the tree associated to the outputs of the model
        """
        return self.treeOutputs
    
    def __SetTrees(self):
        """
        This method sets the trees associated to all parameters, variables, inbputs and outputs
        """
        self.__SetInputsTree()
        self.__SetOutputsTree()
        self.__SetParametersTree()
        self.__SetVariablesTree()
        pass
    
    def __SetParametersTree(self):
        """
        This method updates the parameters tree structure
        """
        if not self.__SetGeneralizedTree(1, None):
            print "Problems while creating the parameters tree"
            self.treeParameters = Tree(Strings.PARAMETER_STRING)
    
    def __SetVariablesTree(self):
        """
        This method updates the variables tree structure
        """
        if not self.__SetGeneralizedTree(3, None, True):
            print "Problems while creating the variables tree"
            self.treeVariables = Tree(Strings.VARIABLE_STRING)
        
    def __SetInputsTree(self):
        """
        This method updates the inputs tree structure
        """
        if not self.__SetGeneralizedTree(None, 0):
            print "Problems while creating the inputs tree"
            self.treeInputs = Tree(Strings.INPUT_STRING)
        
    def __SetOutputsTree(self):
        """
        This method updates the outputs tree structure
        """
        if not self.__SetGeneralizedTree(None, 1):
            print "Problems while creating the outputs tree"
            self.treeOutputs = Tree(Strings.OUTPUT_STRING)
            
    def __SetGeneralizedTree(self, variability, causality, onlyStates = False, pedantic = False):
        """
        This method populates 
        """
        if variability == 1 and causality == None:
            # parameters
            done = self.GetTree(self.treeParameters, variability, causality, onlyStates, pedantic)
        if variability == 3 and causality == None:
            # state variables
            done = self.GetTree(self.treeVariables, variability, causality, onlyStates, pedantic)
        if variability == None and causality == 1:
            # outputs
            done = self.GetTree(self.treeOutputs, variability, causality, onlyStates, pedantic)
        if variability == None and causality == 0:
            # inputs
            done = self.GetTree(self.treeInputs, variability, causality, onlyStates, pedantic)
            
        return done
    
    def __SetInOutVar(self, variability, causality):
        """
        "Input"
            causality = 0
            variability = None
        "Outputs"
            causality = 1
            variability = None
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
            var.SetObject(dictVariables[k])
            
            if variability == None and causality ==0:
                # input
                self.inputs.append(var)
            if variability == None and causality ==1:
                # output
                self.outputs.append(var)             
    
    def __SetInputs(self):
        """
        This function sets the input variables of a model
        """
        self.__SetInOutVar(None, 0)

    def __SetOutputs(self):
        """
        This function sets the output variables of a model
        """
        self.__SetInOutVar(None, 1)
        
    def GetInputByName(self, name):
        """
        This method returns the input contained in the list of inputs that has a name equal to 'name'
        """
        for var in self.inputs:
            if var.GetObject().name == name:
                return var
        return None
    
    def GetOutputByName(self, name):
        """
        This method returns the output contained in the list of outputs that has a name equal to 'name'
        """
        for var in self.outputs:
            if var.GetObject().name == name:
                return var
        return None
    
    def IsParameterPresent(self, object):
        """
        This method returns true is the parameter is contained in the list of parameters of the model
        """
        try:
            self.parameters.index(object)
            return True
        except ValueError:
            # the object is not yet part of the list, add it
            return False
    
    def IsVariablePresent(self, object):
        """
        This method returns true is the variable is contained in the list of variable of the model
        """
        try:
            self.variables.index(object)
            return True
        except ValueError:
            # the object is not yet part of the list, add it
            return False
    
    def AddParameter(self, object):
        """
        This method add one object to the list of parameters. This list contains only the parameters that 
        will be modified during the further analysis
        """
        print "Add parameter: ",object
        if self.IsParameterPresent(object):
            return False
        else:
            # the object is not yet part of the list, add it
            self.parameters.append(object)
            print self.parameters
            return True
    
    def RemoveParameter(self, object):
        """
        This method remove one object to the list of parameters. This list contains only the parameters that 
        will be modified during the further analysis
        """
        try:
            index = self.parameters.index(object)
            self.parameters.pop(index)
            return True
        except ValueError:
            # the object cannot be removed because it is not present
            return False
    
    def ToggleParameter(self, object):
        """
        If the parameter is already present it is removed, otherwise it is added
        """
        if self.IsParameterPresent(object):
            self.RemoveParameter(object)
        else:
            self.AddParameter(object)
    
    def AddVariable(self, object):
        """
        This method add one object to the list of variables. This list contains only the variables that 
        will be modified during the further analysis
        """
        print "Add variable: ",object
        if self.IsVariablePresent(object):
            return False
        else:
            # the object is not yet part of the list, add it
            self.variables.append(object)
            return True
    
    def RemoveVariable(self, object):
        """
        This method remove one object to the list of variables. This list contains only the parameters that 
        will be modified during the further analysis
        """
        try:
            index = self.variables.index(object)
            self.variables.pop(index)
            return True
        except ValueError:
            # the object cannot be removed because it is not present
            return False
    
    def ToggleVariable(self, object):
        """
        If the variable is already present it is removed, otherwise it is added
        """
        if self.IsVariablePresent(object):
            self.RemoveVariable(object)
        else:
            self.AddVariable(object)