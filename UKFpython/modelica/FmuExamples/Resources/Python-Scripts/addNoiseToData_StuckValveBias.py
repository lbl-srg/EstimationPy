import numpy as np
import matplotlib.pyplot as plt
import getCSVdata

# Small bias
#inputFileName = '../data/SimulationData_ValveBias.csv'
#outputFileName = '../data/NoisyData_ValveBias.csv'

# bigger bias
inputFileName = '../data/SimulationData_ValveBias2.csv'
outputFileName = '../data/NoisyData_ValveBias2.csv'

dt = 2.0
(DataMatrix, I, J, csv_writer) = getCSVdata.getCSVdata(inputFileName, outputFileName, dt)

# the columns of the CSV file are
# 0)Time
# 1)valveStuck.m_flow;
# 2)valveStuck.m_flow_real;
# 3)valveStuck.cmd;
# 4)valveStuck.bias;
# 5)valveStuck.lambda;
# 6)valveStuck.dp;
# 7)valveStuck.T_in;
# 8)valveStuck.smoothMax.y

# define the amplitude of the noise for each column
# the noise is uniform and of amplitude +/- Delta_*
Delta_Dp    = 10000.0
Delta_mFlow = 0.05
Delta_T     = 1.0

# compute the error vectors
noise_Dp    = Delta_Dp*(2*np.random.random((I,)) - np.ones((I,)))
noise_mFlow = Delta_mFlow*(2*np.random.random((I,)) - np.ones((I,)))
noise_T     = Delta_T*(2*np.random.random((I,)) - np.ones((I,)))

# create a copy of the original matrix and add the noise
NoiseDataMatrix = DataMatrix.copy()
NoiseDataMatrix[:,1] = NoiseDataMatrix[:,1] + noise_mFlow
NoiseDataMatrix[:,2] = NoiseDataMatrix[:,2] 
NoiseDataMatrix[:,3] = NoiseDataMatrix[:,3] 
NoiseDataMatrix[:,4] = NoiseDataMatrix[:,4]
NoiseDataMatrix[:,6] = NoiseDataMatrix[:,6] + noise_Dp
NoiseDataMatrix[:,7] = NoiseDataMatrix[:,7] + noise_T
NoiseDataMatrix[:,8] = NoiseDataMatrix[:,8]
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
ax1.plot(DataMatrix[:,0],DataMatrix[:,1],'b-', label='$m_{FLOW}$')
ax1.plot(DataMatrix[:,0],NoiseDataMatrix[:,1],'bo')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Mass Flow Rate [kg/s]')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(312)
ax2.plot(DataMatrix[:,0],DataMatrix[:,6],'b-', label='$\Delta P$')
ax2.plot(DataMatrix[:,0],NoiseDataMatrix[:,6],'bo')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Pressure difference [Pa]')
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(313)
ax3.plot(DataMatrix[:,0],DataMatrix[:,3],'g', label='$cmd$')
ax3.plot(DataMatrix[:,0],DataMatrix[:,8],'r', label='$position$')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Actuator position [.]')
ax3.legend()
ax3.grid(True)

plt.show()
