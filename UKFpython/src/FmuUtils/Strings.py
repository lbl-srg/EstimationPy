'''
Created on Nov 6, 2013

@author: marco
'''
# These strings are used to define the main categories of PyFMI variables that can be managed
# for the FMI model
VARIABLE_STRING = "Variables"
PARAMETER_STRING = "Parameters"
INPUT_STRING = "Inputs"
OUTPUT_STRING = "Outputs"

# These strings are used to define the keys of a dictionary that contains the
# simulation results. The string time is the same used by PyFMI to save the simulation
# time vector in the results it provide
TIME_STRING = "time"
DATA_STRING = "data"

# These strings are used to define the three different ways a result can
# be stored by PyFMI
SIMULATION_OPTION_RESHANDLING_STRING = "result_handling"
RESULTS_ON_MEMORY_STRING = "memory"
RESULTS_ON_FILE_STRING = "file"
RESULTS_ON_HANDLER_STRING = "custom"
SIMULATION_OPTION_RESHANDLING_LIST = [RESULTS_ON_MEMORY_STRING, RESULTS_ON_FILE_STRING, RESULTS_ON_HANDLER_STRING]

# These strings are used to identify the names of the numerical solvers available
SOLVER_LSODAR_STRING = "LSODAR"
SOLVER_RUNGEKUTTA34_STRING = "RungeKutta34"
SOLVER_EXPLICITEULET_STRING = "ExplicitEuler"
SOLVER_RODASODE_STRING = "RodasODE"
SOLVER_RADAU_STRING = "Radau5ODE"
SOLVER_CVODE_STRING = "CVode"
SOLVER_DOPRI_STRING = "Dopri5"

SOLVERS_NAMES = [SOLVER_LSODAR_STRING, SOLVER_RUNGEKUTTA34_STRING, SOLVER_EXPLICITEULET_STRING, \
                 SOLVER_EXPLICITEULET_STRING, SOLVER_RODASODE_STRING, SOLVER_RADAU_STRING, \
                 SOLVER_CVODE_STRING, SOLVER_DOPRI_STRING]
SOLVER_NAMES_OPTIONS = [name+"_options" for name in SOLVERS_NAMES]

# These strings are used to identify options of the solvers
SOLVER_OPTION_RTOL_STRING = "rtol"
SOLVER_OPTION_ATOL_STRING = "atol"
SOLVER_OPTION_VERBOSITY_STRING = "verbosity"

# Verbosity levels of the Assimulo solvers
SOLVER_VERBOSITY_QUIET   = 50
SOLVER_VERBOSITY_WHISPER = 40
SOLVER_VERBOSITY_NORMAL  = 30
SOLVER_VERBOSITY_LOUD    = 20
SOLVER_VERBOSITY_SCREAM  = 10
SOLVER_VERBOSITY_LEVELS = [SOLVER_VERBOSITY_QUIET, SOLVER_VERBOSITY_WHISPER, \
                           SOLVER_VERBOSITY_NORMAL, SOLVER_VERBOSITY_LOUD, SOLVER_VERBOSITY_SCREAM]

