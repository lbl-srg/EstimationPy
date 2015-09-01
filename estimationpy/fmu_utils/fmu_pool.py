'''
@author: Marco Bonvini
'''
import multiprocessing
import os
import time
import shutil

from multiprocessing import Process, Queue
from threading import Thread

import estimationpy.fmu_utils.strings as fmu_util_strings

class P(Process):
    """
    This class represents a process running a single simulation of an FMU
    model. Multiple instances of this class can be executed simultaneously to
    run simulations in parallel on multiple processors.
    
    The class has the following attributes:
    
    * ``model``, that is an instance of the class :class:`estimationpy.fmu_utils.model.Model`,
    * ``x0``, the initial state of the model,
    * ``pars``, the parameters to be modified before simulating the model,
    * ``startTime``, the initial time of the simulation period,
    * ``stopTime``, the end time of the simulation period,
    * ``result_queue``, a queue of type :class:`multiprocesing.Queue` where all the results of \
      the simulations are stored and can be retrieved after the simulations are terminated,
    * ``index``, an integer that is used to sort the results data by the class that manages a pool of processes,
    * ``debug``, a boolean flag that enables the debug mode when running the simulations.
    
    """

    def __init__(self, model, x0, pars, startTime, stopTime, results_queue, index, debug = False):
        """
        Constructor of the class initialing the process that runs the simulation.
        
        :param estimationpy.fmu_utils.model.Model model: The model to simulate
        :param numpy.array: the vector containing the initial state of the model
        :param np.array pars: a list or an ordered iterable objects containing the values of the parameters
          that have to be estimated and are defined in the model.
        :param datetime.datetime startTime: the initial time of the simulation period
        :param datetime.datetime stopTime: the end time of the simulation period
        :param multiprocesing.Queue result_queue: the queue that stores the results of the simulation,
        :param int index: the index used to save data in the queue, this is used to identify who generated the
          results during the post processing phase.
        :param bool debug: boolean parameter that specifies if the debug logger is active or not.
        
        """
        super(P, self).__init__()
        self.model = model
        self.x0 = x0
        self.pars = pars
        self.startTime = startTime
        self.stopTime = stopTime
        self.queue = results_queue
        self.index = index
        self.debug = debug
        
    def run(self):
        """
        Method that is called when the :func:`start` method of this class is invoked.
        The method executes the following steps:
        
        1. Sets the values of the selected states,
        2. Sets the values of the parameters selected,
        3. Creates a folder that will contain the results of the simulation in case PyFMI is configured to write to the file system,
        4. Run the simulation by calling the method :func:`estimationpy.fmu_utils.model.Model.simulate`
        5. Saves the results in the queue using the specified index,
        6. Removes the folder containing the result data if they were written.
        
        :return: False, is there are problem during the simulation, None otherwise.
        
        """
        if self.debug:
            print "*"*40
            print "Start simulation"
            print "In process (in pid=%d)...\n" % os.getpid()
        
        # Assign the initial conditions to the states selected
        self.model.set_state_selected(self.x0)
        if self.debug:
            print "Initial condition = "+str(self.model.get_State_observed_values())
            
        # Assign the values to the parameters selected
        self.model.set_parameters_selected(self.pars)
        if self.debug:
            print "Parameters = "+str(self.model.get_parameters_values())
        
        # Check if the options of the model contains the option for writing results to files
        opts = self.model.get_simulation_options()
        workWithFiles = opts[fmu_util_strings.SIMULATION_OPTION_RESHANDLING_STRING] == fmu_util_strings.RESULTS_ON_FILE_STRING
        if workWithFiles:
            # Create an hidden folder named as the Process ID (e.g .4354/)
            dirPath = os.path.join(".","."+str(os.getpid()))
            if not os.path.exists(dirPath):
                    os.makedirs(dirPath)
        
            # Define the name of the file that will contain the results (e.g .4354/results.txt)
            fileName = os.path.join(dirPath,"results.txt")
            self.model.set_result_file(fileName)
    
        # Simulate
        try:
            results = self.model.simulate(start_time = self.startTime, final_time = self.stopTime)
        except Exception, e:
            print str(e)
            print "Problems during simulation"
            results = False
            
        # Put the results in a queue as
        # [index, result]
        # The index will be used to sort the results in the class that manages the processes
        self.queue.put([self.index, results])
    
        # Delete the results contained in the folder
        if workWithFiles:
            shutil.rmtree(dirPath)
        
        return

def threaded_function(queue, results, N_RESULTS):
    """
    This is a function executed in the main thread that reads the values in the queue, 
    and moves them to a dictionary. The function, and thus the thread, terminates when all the
    expected results have been read. The number of expected results is specified by the 
    parameter ``N_RESULTS``.
    
    :param multiprocesing.Queue queue: the queue containing the results generated by the processes.
    :param dict results: reference to a dictionary where the results enqueued are moved.
    :param int N_RESULTS: the number of results to dequeue and move to the dictionary.
    """
    n = 0
    while n < N_RESULTS:
        if not queue.empty():
            temp_res = queue.get()
            # remember that
            # temp_res = [index, results]
            results[temp_res[0]] = temp_res[1:]
            n += 1
    
class FmuPool():
    """
    This class manages a pool of processes that execute parallel simulation
    of an FMU model.
    
    **NOTE:**
    
        The processes running the simulations, executed in parallel if multiple processors are available,
        produce results that are stored in a queue. If the queue reaches its limit the execution will be
        blocked until the resources are freed.
    
    """
    
    def __init__(self, model, processes = multiprocessing.cpu_count()-1, debug = False):
        """
        Constructore that initializes the pool of processes that runs the simulations.
        
        :param estimationpy.fmu_utils.model.Model model: The model to simulate
        :param int processes: the number of processes allocated for the job
        :param bool debug: boolean flag that indicates whether the level of logging
          is for debugging or not. If True a file called ``debugFile.log`` is created.
        """
        self.model = model
        self.debug = debug

        # Set debug flag
        if debug:
            self.f = open('debugFile.log','w')
        
        # Define the number of processes to be used
        if processes >= 1:
            self.N_MAX_PROCESS = processes
        else:
            print "The number of processes specified in a Pool must be >=1"
            self.N_MAX_PROCESS = 1

    def run(self, values, start = None, stop = None):
        """
        This method performs the simulation of the model with multiple initial states or
        parameters using multiple processes in parallel.
        The parameter ``values`` is a list that contains the values that the states and parameters
        will assume in the multiple simulations.
        For example::
        
            pars = [{"state":[1,1,1], "parameters": [0,1,2]},
                    {"state":[1,1,10], "parameters": [0,2,2]},
                    {"state":[1,1,100], "parameters": [0,3,2]},
                    {"state":[1,1,1000], "parameters": [0,4,2]}]
        
        indicates to run four simulations in parallel, and for each simulation the values of the initial
        states and parameters are the ones indicated by the dictionary.
        
        :param list values: a list of dictionaries that contains the values of the initial states and
          parameters used in each of the simulations.
        :param datetime.datetime start: the initial time for the simulation, if not specified the initial time
          of the data series associated to the inputs of the models is used
        :param datetime.datetime stop: the final time for the simulation, if not specified the final time
          of the data series associated to the inputs of the models is used
        
        :return: a dictionary that contains the results of each simulation. The results are indexed with integers that
          correspond to the positions of the elements in ``pars``. For example ``results[0]`` contains the
          results of the simulation run with state and parameters specified by ``pars[0]["state"]`` and ``pars[0]["parameters"]``.
          If a problem occurs when running the simulations, an empty dictionary is returned.
        :rtype: dict
        
        """
        # Define a Queue of results
        results_queue = Queue()
        # Define a list of processes
        processes = []
        
        # number of simulations to perform
        N_SIMULATIONS = len(values)
    
        j = 0
        for v in values:
            # For every initial value a different simulation has to be run
            # Initialize a process that will perform the simulation
            x0 = v["state"]
            pars = v["parameters"]
            p = P(self.model, x0, pars, start, stop, results_queue, j, self.debug)

            # Append the process to the list
            processes.append(p)
        
            # Increment the index, that is passed to the process
            # this will be useful to order the results (since the processes may end
            # not in order and thus the results will be pushed in the queue in an arbitrary way)
            j += 1

        # Create a dictionary that will contain the results of each simulation.
        # It is a dictionary with as key value the index, and as value the results obtained by the simulation
        results = {}

        # Create a Thread in the main process that will read the data from the queue and put them into the
        # dictionary previously defined.
        # N.B. The Thread will remove elements from the queue right after they have been produced,
        # otherwise the queue will reach the size limit and block the processes running the simulations
        thread = Thread(target = threaded_function, args = (results_queue, results, N_SIMULATIONS ))
        thread.daemon = True
        thread.start()
    
        # Start the process in parallel. This loop maintain the number of active processes at a given limit
        # specified by the 'N_MAX_PROCESS'
        i = 0
        n_active = 0
        finished = False

        # Start measuring the time
        T0 = time.time()

        # The while loop will end when every process associated to a simulation task have been run
        # AND they terminated
        while not finished:

            while n_active < self.N_MAX_PROCESS and i < N_SIMULATIONS:
                # Run the process that simulate
                if self.N_MAX_PROCESS <= 1:
                    # Just one process to run, void to do a fork
                    # NOTE: This is used when the process runs with Celery
                    processes[i].run()
                else:
                    # More than one process that can be run in parallel, spawn a new process
                    processes[i].start()
                    
                i += 1
                if self.debug:
                    self.f.write('Process '+str(processes[i-1].pid)+' Started ('+str(i)+'/'+str(N_SIMULATIONS)+') \n')
            
                # Check how many active processes have been run
                n_active = len(multiprocessing.active_children())
                if self.debug:
                    self.f.write('N_process_active '+str(n_active)+'\n')
                    self.f.write('Queue has size '+str(results_queue.qsize())+'\n')
        
            # Wait the end of the processes to run others otherwise to exit the loop
            # N.B. 'multiprocessing.active_children()' call a .join() to every children already terminated
            n_active = len(multiprocessing.active_children())

            # This condition ensure that the while loop is left when all the process have been terminated
            finished = True if n_active==0 and i == N_SIMULATIONS else False 
    
        # Wait for the thread that all the data created by the processes 
        thread.join(0.5)

        # Stop Measuring the time
        Tend = time.time()
        
        if self.debug:
            # print the time spent for the simulations
            print "\n"+"="*52
            print "Time spent for "+str(N_SIMULATIONS)+" simulations = "+str(Tend - T0)+" [s]"
            print "="*52

        if self.debug:
            self.f.close()
        
        # Create an empty list of results, and put the elements of the dictionary in order
        try:
            res = []
            for k in range(N_SIMULATIONS):
                res.insert(k, results[k])
        except KeyError:
            print "ERROR-- Problems while collecting the results generated by the pool of workers"
            print "number of simulations run in the e pool is", N_SIMULATIONS
            print "number of available results is", len(results)
            res = {}
        
        # return the list of results
        return res
