'''
Created on Sep 19, 2013

@author: marco
'''
import csv
import numpy as np

def getCsvData(inputFileName):
    
    # open the csv file and instantiate the csv reader
    file_in  = open(inputFileName)
    csv_reader = csv.reader(file_in)
    
    rows = 0
    N = 0
    for line in csv_reader:
        if rows==0 :
            # Read the header
            header = line
            N = len(header)
        else :
            r = np.zeros(N).astype(np.float)
            j = 0    
            for item in line:
                r[j] = np.longdouble(item)
                j += 1
    
            if rows == 1:
                DataMatrix = r
            else:
                DataMatrix = np.vstack((DataMatrix,r)).astype(np.float)
        rows += 1
    
    return DataMatrix