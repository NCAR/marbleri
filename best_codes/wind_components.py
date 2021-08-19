import numpy as np
#
# Calculate wind components in polar coordinates
def wc(x, y, u, v):
	#
	# Get theta
	theta = x * 0
	index_sector = np.logical_and((x >= 0), (y >= 0))
	theta[index_sector] = np.arctan(y[index_sector] / x[index_sector])
	index_sector = np.logical_and((x < 0), (y >= 0))
	theta[index_sector] = (np.pi - np.arctan(y[index_sector] / 
		abs(x[index_sector])))
	index_sector = np.logical_and((x < 0), (y < 0))
	theta[index_sector] = (np.pi + np.arctan(abs(y[index_sector]) / 
		abs(x[index_sector])))
	index_sector = np.logical_and((x >= 0), (y < 0))
	theta[index_sector] = (2.0 * np.pi - np.arctan(abs(y[index_sector]) / 
		x[index_sector]))
	#
	# Calculate v_r and v_th components of the wind
	v_tan = v * np.cos(theta) - u * np.sin(theta)
	u_rad = u * np.cos(theta) + v * np.sin(theta)
	return u_rad, v_tan
