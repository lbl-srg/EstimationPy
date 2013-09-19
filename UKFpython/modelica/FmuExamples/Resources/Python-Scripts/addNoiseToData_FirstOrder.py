import numpy as np
import matplotlib.pyplot as plt
import getCSVdata

inputFileName = '../data/SimulationData_FirstOrder.csv'
outputFileName = '../data/NoisySimulationData_FirstOrder.csv'

dt = 1.0
(DataMatrix, I, J, csv_writer) = getCSVdata.getCSVdata(inputFileName, outputFileName, dt)

# the columns of the CSV file are
# 1)Time,
# 2)system.u,
# 3)system.x,
# 4)system.y

# define the amplitude of the noise for each column
# the noise is uniform and of amplitude +/- Delta_*
Delta_u = 0.3
Delta_y = 1.0

# compute the error vectors
noise_u = Delta_u*(2*np.random.random((I,)) - np.ones((I,)))
noise_y = Delta_y*(2*np.random.random((I,)) - np.ones((I,)))

# create a copy of the original matrix and add the noise
NoiseDataMatrix = DataMatrix.copy()
NoiseDataMatrix[:,1] = NoiseDataMatrix[:,1] + noise_u
NoiseDataMatrix[:,3] = NoiseDataMatrix[:,3] + noise_y
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
ax1.plot(DataMatrix[:,0],DataMatrix[:,1],'b-', label='$u$')
ax1.plot(DataMatrix[:,0],NoiseDataMatrix[:,1],'bo')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Input')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(212)
ax2.plot(DataMatrix[:,0],DataMatrix[:,3],'g-', label='$y$')
ax2.plot(DataMatrix[:,0],NoiseDataMatrix[:,3],'go')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Output')
ax2.legend()
ax2.grid(True)

plt.show()

