import numpy as np
import matplotlib.pyplot as plt
import getCSVdata

def compareData(ref, x, thresholds):
	conditionOver = x > ref+thresholds
	conditionBelow = x < ref-thresholds
	if conditionOver.any() or conditionBelow.any():
		return False
	else:
		return True  

inputFileName = '../data/DataPump_16to19_Oct2012.csv'
outputFileName = '../data/DataPump_16to19_Oct2012_variableStep.csv'

(DataMatrix, I, J, csv_writer) = getCSVdata.getCSVdata(inputFileName, outputFileName)

# the columns of the CSV file are
# 1)Time,
# 2)Pump.Speed,
# 3)Pump.kW,
# 4)Pump.gpm

# Define the noise amplitude levels for each meaured data series
Noise_speed = 0.5
Noise_kW = 0.05
Noise_gpm = 6
thresh = np.array([Noise_speed, Noise_kW, Noise_gpm])
newDataMatrix = []

i = 0
skipping = False
old_row = []
for row in DataMatrix:
	t = row[0]
	data = row[1:]
	if i==0:
		t_ref = t
		d_ref = data
		newDataMatrix.append(row)
	else:
		if compareData(d_ref, data, thresh):
			# there is no variation
			skipping = True
		else:
			# there is a variation
			t_ref = t
			d_ref = data
			if skipping:
				newDataMatrix.append(old_row)
			newDataMatrix.append(row)
			skipping = False
	
	i += 1
	old_row = row

# Be sure it is a numpy array	
newDataMatrix = np.array(newDataMatrix)

print "Original form data:",np.shape(DataMatrix)
print "Modified form data:",np.shape(newDataMatrix)

# write data to CSV file
for row in newDataMatrix:
	csv_writer.writerow(row)
print "Variable step sampling done..."

print "\nPlotting..."
# plot the figures that show the difference between the simulation data
# and the data corrupted by noise
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(DataMatrix[:,0],DataMatrix[:,1],'b.', label='$rpm$')
ax1.plot(newDataMatrix[:,0],newDataMatrix[:,1],'b-', label='$rpm^{NEW}$')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Speed')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(312)
ax2.plot(DataMatrix[:,0],DataMatrix[:,2],'g.', label='$P_{El}$')
ax2.plot(newDataMatrix[:,0],newDataMatrix[:,2],'g-', label='$P_{El}^{NEW}$')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('kW')
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(313)
ax3.plot(DataMatrix[:,0],DataMatrix[:,3],'r.', label='$\dot{m}$')
ax3.plot(newDataMatrix[:,0],newDataMatrix[:,3],'r-', label='$\dot{m}^{NEW}$')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('gpm')
ax3.legend()
ax3.grid(True)

plt.show()

