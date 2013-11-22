import numpy as np
import matplotlib.pyplot as plt
import getCSVdata

inputFileName = '../data/SimulationData_StuckValve.csv'
outputFileName = '../data/NoisySimulationData_StuckValve.csv'

dt = 5.0
(DataMatrix, I, J, csv_writer) = getCSVdata.getCSVdata(inputFileName, outputFileName, dt)

# the columns of the CSV file are
# 1)Time,
# 2)valveStuck.Kv,
# 3)valveStuck.cmd,
# 4)valveStuck.dp,
# 5)valveStuck.m_flow,
# 6)valveStuck.leakPosition,
# 7)valveStuck.stuckPosition,
# 8)valveStuck.limiter.y

# define the amplitude of the noise for each column
# the noise is uniform and of amplitude +/- Delta_*
Delta_Dp    = 10000.0
Delta_mFlow = 0.035

# compute the error vectors
noise_Dp  = Delta_Dp*(2*np.random.random((I,)) - np.ones((I,)))
noise_mFlow = Delta_mFlow*(2*np.random.random((I,)) - np.ones((I,)))

# create a copy of the original matrix and add the noise
NoiseDataMatrix = DataMatrix.copy()
NoiseDataMatrix[:,1] = NoiseDataMatrix[:,1]
NoiseDataMatrix[:,2] = NoiseDataMatrix[:,2]
NoiseDataMatrix[:,3] = NoiseDataMatrix[:,3] + noise_Dp
NoiseDataMatrix[:,4] = NoiseDataMatrix[:,4] + noise_mFlow
NoiseDataMatrix[:,6] = NoiseDataMatrix[:,6]
NoiseDataMatrix[:,7] = NoiseDataMatrix[:,7]
print "\nComputed the noise to add..."

# write data to CSV file
for i in range(I):
	csv_writer.writerow(NoiseDataMatrix[i,:])
print "Noise added"

print "\nPlotting..."
# plot the figures that show the difference between the simulation data
# and the data corrupted by noise
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(DataMatrix[:,0],DataMatrix[:,4],'b-', label='$m_{FLOW}$')
ax1.plot(DataMatrix[:,0],NoiseDataMatrix[:,4],'bo')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Mass Flow Rate [kg/s]')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(312)
ax2.plot(DataMatrix[:,0],DataMatrix[:,3],'b-', label='$\Delta P$')
ax2.plot(DataMatrix[:,0],NoiseDataMatrix[:,3],'bo')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Pressure difference [Pa]')
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(313)
ax3.plot(DataMatrix[:,0],DataMatrix[:,2],'g', label='$cmd$')
ax3.plot(DataMatrix[:,0],DataMatrix[:,7],'r', label='$position$')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Actuator position [.]')
ax3.legend()
ax3.grid(True)

plt.show()
