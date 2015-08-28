'''
@author: Marco Bonvini

This module defines constant that can be conveniently
reached from the other modules.

Examples of constants that are defined are

* strings used to define the main categories of PyFMI variables that can be managed,

* strings used to identify data and time in dictionaries that store results
  of simulations,

* the solvers supported by PyFMI (LSODAR, RungeKutta34, RodasODE, etc.),

* the options for the solvers such as absolute and relative tolerances, and verbosity
  of the logger.

  
'''

# These strings are used to define the main categories of PyFMI variables that can be managed
# for the FMI model
VARIABLE_STRING = "Variables"
"""String that identifies a generic variable"""
PARAMETER_STRING = "Parameters"
"""String that identifies a parameter"""
INPUT_STRING = "Inputs"
"""String that identifies an input variable"""
OUTPUT_STRING = "Outputs"
"""String that identifies an output variable"""

# These strings are used to define the keys of a dictionary that contains the
# simulation results. The string time is the same used by PyFMI to save the simulation
# time vector in the results it provide
TIME_STRING = "time"
"""Keywork that identifies time in a dictionary that collects multiple simulation results"""
DATA_STRING = "data"
"""Keywork that identifies the data in a dictionary that collects multiple simulation results"""

# These strings are used to define the three different ways a result can
# be stored by PyFMI
SIMULATION_OPTION_RESHANDLING_STRING = "result_handling"
"""Keyword that specifies how PyFMI should hanle simulation results"""

RESULTS_ON_MEMORY_STRING = "memory"
"""String that specifies that simulation results computed by PyFMI should be handled in memory"""

RESULTS_ON_FILE_STRING = "file"
"""String that specifies that simulation results computed by PyFMI should be handled with files"""

RESULTS_ON_HANDLER_STRING = "custom"
"""String that specifies that simulation results computed by PyFMI should be handled in a custom way"""

SIMULATION_OPTION_RESHANDLING_LIST = [RESULTS_ON_MEMORY_STRING, RESULTS_ON_FILE_STRING, RESULTS_ON_HANDLER_STRING]
"""List containing the different types of result handling available"""

# These strings are used to identify the names of the numerical solvers available
SOLVER_LSODAR_STRING = "LSODAR"
"""String that identifies the LSODAR solver"""

SOLVER_RUNGEKUTTA34_STRING = "RungeKutta34"
"""String that identifies the Runge Kutta 3-4 explicit solver"""

SOLVER_EXPLICITEULET_STRING = "ExplicitEuler"
"""String that identifies the explicit euler solver"""

SOLVER_RODASODE_STRING = "RodasODE"
"""String that identifies the Rodas ODE solver"""

SOLVER_RADAU_STRING = "Radau5ODE"
"""String that identifies the Radau 5th order ODE solver"""

SOLVER_CVODE_STRING = "CVode"
"""String that identifies the CV ODE solver"""

SOLVER_DOPRI_STRING = "Dopri5"
"""String that identifies the Dorman-Price solver"""


SOLVERS_NAMES = [SOLVER_LSODAR_STRING, SOLVER_RUNGEKUTTA34_STRING, SOLVER_EXPLICITEULET_STRING, \
                 SOLVER_EXPLICITEULET_STRING, SOLVER_RODASODE_STRING, SOLVER_RADAU_STRING, \
                 SOLVER_CVODE_STRING, SOLVER_DOPRI_STRING]
"""List containing the names of the solvers available"""

SOLVER_NAMES_OPTIONS = [name+"_options" for name in SOLVERS_NAMES]
"""List containing the keywords identifying the options for a specific solver"""

# These strings are used to identify options of the solvers
SOLVER_OPTION_RTOL_STRING = "rtol"
"""Keywork that identifies the relative tolerance of a solver"""

SOLVER_OPTION_ATOL_STRING = "atol"
"""Keywork that identifies the absolute tolerance of a solver"""

SOLVER_OPTION_VERBOSITY_STRING = "verbosity"
"""Keyword that identify the verbosity used by the solver"""

# Verbosity levels of the Assimulo solvers
SOLVER_VERBOSITY_QUIET   = 50
"""Value that identifies the verbosity to be quiet"""
SOLVER_VERBOSITY_WHISPER = 40
"""Value that identifies the verbosity to be whisper"""
SOLVER_VERBOSITY_NORMAL  = 30
"""Value that identifies the verbosity to be normal"""
SOLVER_VERBOSITY_LOUD    = 20
"""Value that identifies the verbosity to be loud"""
SOLVER_VERBOSITY_SCREAM  = 10
"""Value that identifies the verbosity to be scream"""

SOLVER_VERBOSITY_LEVELS = [SOLVER_VERBOSITY_QUIET, SOLVER_VERBOSITY_WHISPER, \
                           SOLVER_VERBOSITY_NORMAL, SOLVER_VERBOSITY_LOUD, SOLVER_VERBOSITY_SCREAM]
"""List with verbosity levels from quiet to scream"""

