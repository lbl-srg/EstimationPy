import csv
import numpy as np

def getCSVdata(inputFileName, outputFileName, samplingTime = 0.0):
	
	# open the csv file and instantiate the csv reader
	file_in  = open(inputFileName)
	csv_reader = csv.reader(file_in)
	print "Open file: "+str(inputFileName)
	
	# open the csv output file and delete it if already existing
	# instantiate the csv writer
	file_out = open(outputFileName,'w')
	csv_writer = csv.writer(file_out)
	print "Created file: "+str(outputFileName)
	
	rows = 0
	N = 0
	for line in csv_reader:
		if rows==0 :
			# Read the header
			header = line
			N = len(header)
	
			# copy the header in the output csv file
			csv_writer.writerow(header)
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
	
	I, J = np.shape(DataMatrix)
	print "Finished reading the input CSV file,"
	print "it has: "+str(I)+" rows and "+str(J)+" columns"
	
	if samplingTime > 0.0:
		print "Re-sampling of the data obtained with simulation is required"
		print "Time step is DT = "+str(samplingTime)
		
		time = DataMatrix[:,0]
		t_start = time[0]
		t_stop  = time[-1]
		
		newTime = np.arange(t_start, t_stop+0.1, samplingTime)
		numSamples = len(newTime)
		
		interpDataMatrix      = np.zeros((numSamples,J))
		interpDataMatrix[:,0] = newTime
		
		for j in range(1,J):
			interpDataMatrix[:,j] = np.interp(newTime, time, DataMatrix[:,j])
		
		print "Finished re-sampling of the data,"
		print "Now there are: "+str(numSamples)+" rows and "+str(J)+" columns"
		
		return (interpDataMatrix, numSamples, J, csv_writer)
	else:
		return (DataMatrix, I, J, csv_writer)