import csv
import numpy as np
import matplotlib.pyplot as plt

inputFileName = '../data/SimulationData.csv'
outputFileName = '../data/NoisySimulationData.csv'

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
		r = np.zeros(N).astype(np.longdouble)
		j = 0	
		for item in line:
			r[j] = np.longdouble(item)
			#print "String "+str(item)+" => "+str(r[j])
			j += 1

		if rows == 1:
			DataMatrix = r
		else:
			DataMatrix = np.vstack((DataMatrix,r)).astype(np.longdouble)
	rows += 1

I, J = np.shape(DataMatrix)
print "Finished reading the input CSV file,"
print "it has: "+str(I)+" rows and "+str(J)+" columns"

# the columns of the CSV file are
# 1)Time,
# 2)heatExchanger.Tcold_IN,
# 3)heatExchanger.Tcold_OUT,
# 4)heatExchanger.Thot_IN,
# 5)heatExchanger.Thot_OUT,
# 6)heatExchanger.metal.T,
# 7)heatExchanger.mFlow_HOT,
# 8)heatExchanger.mFlow_COLD

# define the amplitude of the noise for each column
# the noise is uniform and of amplitude +/- Delta_*
Delta_TcIN  = 1.0
Delta_TcOUT = 1.0
Delta_ThIN  = 1.0
Delta_ThOUT = 1.0
Delta_mc    = 0.02
Delta_mh    = 0.02

# compute the error vectors
noise_TcIN  = Delta_TcIN*(2*np.random.random((I,)) - np.ones((I,)))
noise_TcOUT = Delta_TcOUT*(2*np.random.random((I,)) - np.ones((I,)))
noise_ThIN  = Delta_ThIN*(2*np.random.random((I,)) - np.ones((I,)))
noise_ThOUT = Delta_ThOUT*(2*np.random.random((I,)) - np.ones((I,)))
noise_mc    = Delta_mc*(2*np.random.random((I,)) - np.ones((I,)))
noise_mh    = Delta_mh*(2*np.random.random((I,)) - np.ones((I,)))

# create a copy of the original matrix and add the noise
NoiseDataMatrix = DataMatrix.copy()
NoiseDataMatrix[:,1] = NoiseDataMatrix[:,1] + noise_TcIN
NoiseDataMatrix[:,2] = NoiseDataMatrix[:,2] + noise_TcOUT
NoiseDataMatrix[:,3] = NoiseDataMatrix[:,3] + noise_ThIN
NoiseDataMatrix[:,4] = NoiseDataMatrix[:,4] + noise_ThOUT
NoiseDataMatrix[:,6] = NoiseDataMatrix[:,6] + noise_mh
NoiseDataMatrix[:,7] = NoiseDataMatrix[:,7] + noise_mc
print "\nComputed the noise to add..."

# write data to CSV file
for i in range(I):
	csv_writer.writerow(NoiseDataMatrix[i,:])
print "Noise added"

print "\nPlotting..."
# plot the figures that show the difference between the simulation data
# and the data corrupted by noise
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(DataMatrix[:,0],DataMatrix[:,1],'b-', label='$T_{COLD}^{IN}$')
ax1.plot(DataMatrix[:,0],NoiseDataMatrix[:,1],'bo')
ax1.plot(DataMatrix[:,0],DataMatrix[:,2],'g-', label='$T_{COLD}^{OUT}$')
ax1.plot(DataMatrix[:,0],NoiseDataMatrix[:,2],'go')
ax1.plot(DataMatrix[:,0],DataMatrix[:,3],'r-', label='$T_{HOT}^{IN}$')
ax1.plot(DataMatrix[:,0],NoiseDataMatrix[:,3],'ro')
ax1.plot(DataMatrix[:,0],DataMatrix[:,4],'k-', label='$T_{HOT}^{OUT}$')
ax1.plot(DataMatrix[:,0],NoiseDataMatrix[:,4],'ko')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Temperatures [K]')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(212)
ax2.plot(DataMatrix[:,0],DataMatrix[:,6],'b-', label='$\dot{m}_{HOT}$')
ax2.plot(DataMatrix[:,0],NoiseDataMatrix[:,6],'bo')
ax2.plot(DataMatrix[:,0],DataMatrix[:,7],'g-', label='$\dot{m}_{COLD}$')
ax2.plot(DataMatrix[:,0],NoiseDataMatrix[:,7],'go')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Mass flow rates [kg/s]')
ax2.legend()
ax2.grid(True)

plt.show()

