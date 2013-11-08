'''
Created on Oct 17, 2013

@author: marco
'''
import multiprocessing
import os
import time
import shutil

from multiprocessing import Process, Queue
from threading import Thread


class P(Process):
    """
    This class represents a single process running a simulation
    """

    def __init__(self, model, x0, startTime, stopTime, results_queue, index):
        """
        Initialization method of the process that runs a simulation.
        """
        super(P, self).__init__()
        self.model = model
        self.x0 = x0
        self.startTime = startTime
        self.stopTime = stopTime
        self.queue = results_queue
        self.index = index

    def run(self):
        """
        This method will be called by the .start()  method of the process
        """
        
        print "*"*40
        print "Start simulation"
        print "In process (in pid=%d)...\n" % os.getpid()
        
        # Assign the initial conditions to the model
        self.model.SetState(self.x0)
        print "Initial condition = "+str(self.model.GetState())
        
        # Create an hidden folder named as the Process ID (e.g .4354/)
        dirPath = os.path.join(".","."+str(os.getpid()))
        if not os.path.exists(dirPath):
                os.makedirs(dirPath)
    
        # Define the name of the file that will contain the results (e.g .4354/results.txt)
        fileName = os.path.join(dirPath,"results.txt")
        self.model.SetResultFile(fileName)
    
        # Simulate
        results = self.model.Simulate(start_time = self.startTime, final_time = self.stopTime)
    
        # Put the results in a queue as
        # [index, result]
        # The index will be used to sort the results in the class that manages the processes
        self.queue.put([self.index, results])
    
        # Delete the results contained in the folder
        shutil.rmtree(dirPath)
        
        return

def threaded_function(queue, results, N_RESULTS):
    """
    The processes running the simulations (executed in parallel) produce results that are stored in
    a queue. If this queue reaches its limit the execution will be blocked.
    
    A thread will execute this function that reads the values in the queue, and move them to a dictionary.
    The thread ends when all the results have been read.
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
    This class represents a pool of processes that will run the simulations in parallel
    """
    
    def __init__(self, model, processes = multiprocessing.cpu_count(), debug = False):
        """
        Initialization of the pool of processes that will run the simulations
        """
        self.model = model
        self.debug = debug

        # Set debug flag
        if debug:
            self.f = open('debugFile.log','w')
        
        # Define the number of processors to be used
        if processes >= 1:
            self.N_MAX_PROCESS = processes
        else:
            print "The number of processes specified in a Pool must be >=1"
            self.N_MAX_PROCESS = 1

    def Run(self, initValues, start = None, stop = None):
        """
        This method runs the multiple simulations across the processes
        """
        # Define a Queue of results
        results_queue = Queue()
        # Define a list of processes
        processes = []
        
        # number of simulations to perform
        N_SIMULATIONS = len(initValues)
    
        j = 0
        for x0 in initValues:
            # For every initial value a different simulation has to be run
            # Initialize a process that will perform the simulation
            p = P(self.model, x0, start, stop, results_queue, j)

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
    
        # print the time spent for the simulations
        print "\n"+"="*52
        print "Time spent for "+str(N_SIMULATIONS)+" simulations = "+str(Tend - T0)+" [s]"
        print "="*52

        if self.debug:
            self.f.close()
        
        # Create an empty list of results, and put the elements of the dictionary in order
        res = []
        for k in range(N_SIMULATIONS):
            res.insert(k, results[k])
        
        # return the list of results
        return res