import numpy as np
import matplotlib.pyplot as plt
import getCSVdata

# Small bias
#inputFileName = '../data/SimulationData_ValveBias.csv'
#outputFileName = '../data/NoisyData_ValveBias.csv'

# No bias bias
#inputFileName = '../data/SimulationData_ValveBias2.csv'
#outputFileName = '../data/NoisyData_ValveBias2.csv'

# No bias bias, new parameters
#inputFileName = '../data/SimulationData_ValveBias3.csv'
#outputFileName = '../data/NoisyData_ValveBias3.csv'

# No bias bias, new parameters, long experiment
inputFileName = '../data/SimulationData_ValveBias4.csv'
outputFileName = '../data/NoisyData_ValveBias4.csv'

dt = 3.0
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
# 8)valveStuck.valve.opening

# define the amplitude of the noise for each column
# the noise is uniform and of amplitude +/- Delta_*
Delta_Dp    = 2500.0
Delta_mFlow = 0.02
Delta_T     = 0.8

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
ax1 = fig.add_subplot(411)
ax1.plot(DataMatrix[:,0],DataMatrix[:,1],'b-', label='$m_{FLOW}$')
ax1.plot(DataMatrix[:,0],DataMatrix[:,2],'g', label='$m_{FLOW}^{Real}$')
ax1.plot(DataMatrix[:,0],NoiseDataMatrix[:,1],'bo')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Mass Flow Rate [kg/s]')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(412)
ax2.plot(DataMatrix[:,0],DataMatrix[:,6],'b-', label='$\Delta P$')
ax2.plot(DataMatrix[:,0],NoiseDataMatrix[:,6],'bo')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Pressure difference [Pa]')
ax2.legend()
ax2.grid(True)

ax2 = fig.add_subplot(413)
ax2.plot(DataMatrix[:,0],DataMatrix[:,7],'r-', label='$T$')
ax2.plot(DataMatrix[:,0],NoiseDataMatrix[:,7],'ro')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('temperature [K]')
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(414)
ax3.plot(DataMatrix[:,0],DataMatrix[:,3],'g', label='$cmd$')
ax3.plot(DataMatrix[:,0],DataMatrix[:,8],'r', label='$position$')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Actuator position [.]')
ax3.legend()
ax3.grid(True)

plt.show()
