#
# Python script to rotate forecast fields such that the 
#  shear vector is pointing east
#
import matplotlib.pyplot as plt
from netCDF4 import Dataset, chartostring
import numpy as np
import sys
from scipy import ndimage
from scipy.interpolate import griddata
from vortz_calc import vortz
from wind_components import wc
import time
start_time = time.time()
#
# Input from parent program
basedir_input = sys.argv[1]
basedir_input_raw = sys.argv[2]
basedir_output = sys.argv[3]
input_file = sys.argv[4]
#input_file = 'matthew14l.2016100312.f024.nc'
file_out = basedir_output + input_file
#
inc_skip = 2
#
print(' ')
print('Program to rotate HWRF fields beginning')

pred_dir = '/glade/p/ral/nsap/rozoff/hfip/besttrack_predictors/'
#
# Read in data
#  -> Predictor datasets
file_in = pred_dir + 'diag_2015_2017_adecks_atl_bug_corrected.nc'
f = Dataset(file_in, 'r', format = 'NETCDF4')
shr_hdg = f.variables['SHR_HDG'][:]
latc = f.variables['LAT'][:]
lonc = f.variables['LON'][:]
date = chartostring(f.variables['DATE'][:])
stnum = chartostring(f.variables['STNUM'][:])
stnum = np.asarray([int(stnum_piece) for stnum_piece in stnum])
N = len(stnum)
basin = ['l'] * N
f.close
#
file_in = pred_dir + 'diag_2015_2017_adecks_epo_bug_corrected.nc'
f = Dataset(file_in, 'r', format = 'NETCDF4')
shr_hdg = np.concatenate((shr_hdg, f.variables['SHR_HDG'][:]), axis = 1)
latc = np.concatenate((latc, f.variables['LAT'][:]), axis = 1)
lonc = np.concatenate((lonc, f.variables['LON'][:]), axis = 1)
date = np.concatenate((date, chartostring(f.variables['DATE'][:])), axis = 0)
stnum_temp = chartostring(f.variables['STNUM'][:])
stnum_temp = np.asarray([int(stnum_piece) for stnum_piece in stnum_temp])
stnum = np.concatenate((stnum, stnum_temp), axis = 0)
N = len(stnum_temp)
basin = np.concatenate((basin, ['e'] * N), axis = 0)
f.close
#
file_in = pred_dir + 'diag_2018_atl_bug_corrected.nc'
f = Dataset(file_in, 'r', format = 'NETCDF4')
shr_hdg = np.concatenate((shr_hdg, f.variables['SHR_HDG'][:]), axis = 1)
latc = np.concatenate((latc, f.variables['LAT'][:]), axis = 1)
lonc = np.concatenate((lonc, f.variables['LON'][:]), axis = 1)
date = np.concatenate((date, chartostring(f.variables['DATE'][:])), axis = 0)
stnum_temp = chartostring(f.variables['STNUM'][:])
stnum_temp = np.asarray([int(stnum_piece) for stnum_piece in stnum_temp])
stnum = np.concatenate((stnum, stnum_temp), axis = 0)
N = len(stnum_temp)
basin = np.concatenate((basin, ['l'] * N), axis = 0)
f.close
#
file_in = pred_dir + 'diag_2018_epo_bug_corrected.nc'
f = Dataset(file_in, 'r', format = 'NETCDF4')
shr_hdg = np.concatenate((shr_hdg, f.variables['SHR_HDG'][:]), axis = 1)
latc = np.concatenate((latc, f.variables['LAT'][:]), axis = 1)
lonc = np.concatenate((lonc, f.variables['LON'][:]), axis = 1)
date = np.concatenate((date, chartostring(f.variables['DATE'][:])), axis = 0)
stnum_temp = chartostring(f.variables['STNUM'][:])
stnum_temp = np.asarray([int(stnum_piece) for stnum_piece in stnum_temp])
stnum = np.concatenate((stnum, stnum_temp), axis = 0)
N = len(stnum_temp)
basin = np.concatenate((basin, ['e'] * N), axis = 0)
f.close
#
# Files associated with the 2D HWRF fields (input and output)
file_in = basedir_input + input_file
file_in_raw = basedir_input_raw + input_file
file_out = basedir_output + input_file
# 
# Get the already derived Theta_e data
f = Dataset(file_in, 'r', format = 'NETCDF4')
th_p = f.variables['THETA_E_L100'][:]
th_2m = f.variables['THETA_E_L103'][:]
lat = f.variables['lat_0'][:]
lon = f.variables['lon_0'][:] - 360.0
lvls = f.variables['lv_ISBL0'][:]
f.close
#
# Get rid of the 700-hPa level (cv = cut vertical)
ind_cv = np.where(lvls != 70000.0)
lvls = lvls[ind_cv]
th_p = np.squeeze(th_p[ind_cv, :, :])
#
# Read in raw wind data for center finding adjustment
f = Dataset(file_in_raw, 'r', format = 'NETCDF4')
u_raw = f.variables['UGRD_P0_L103_GLL0'][:]
v_raw = f.variables['VGRD_P0_L103_GLL0'][:]
u_p = f.variables['UGRD_P0_L100_GLL0'][:]
v_p = f.variables['VGRD_P0_L100_GLL0'][:]
tcolc = f.variables['TCOLC_P0_L200_GLL0'][:]
#
u_p = np.squeeze(u_p[ind_cv, :, :])
v_p = np.squeeze(v_p[ind_cv, :, :])
#
for name, variable in f.variables.items():
	initial_time = getattr(variable, 'initial_time')
	forecast_time = getattr(variable, 'forecast_time')
	break
f.close()
#
# Get file identification information
mo_stm = int(initial_time[0:2])
da_stm = int(initial_time[3:5])
yr_stm = int(initial_time[6:10])
hr_stm = int(initial_time[12:14])
date_stm = str(initial_time[6:10] + initial_time[0:2] + initial_time[3:5] + initial_time[12:14])
lt_stm = forecast_time
lt_stm_idx = int(lt_stm / 3)
# 
# Match basin 
result = [pos for pos, char in enumerate(input_file) if char == '.']
stnum_stm = int(input_file[result[0]-3:result[0]-1])
basin_stm = input_file[result[0]-1]
basin_res = [i for i in basin if basin_stm in i]
indices_basin = [i for i, s in enumerate(basin) if basin_stm in s]
if not indices_basin:
 print('No matching basin found. Skipping.')
 sys.exit()
#
stnum = stnum[indices_basin]
shr_hdg = shr_hdg[:, indices_basin]
latc = latc[:, indices_basin]
lonc = lonc[:, indices_basin]
date = date[indices_basin]
#
# Match date
indices_date = [i for i, s in enumerate(date) if date_stm in s]
if not indices_date:
 print('No matching dates found. Skipping.')
 sys.exit()
date = [i for i in date if date_stm in i]
stnum = stnum[indices_date]
shr_hdg = shr_hdg[:, indices_date]
latc = latc[:, indices_date]
lonc = lonc[:, indices_date]
#
# Match storm number
indices_stnum = np.where(stnum == stnum_stm)
indices_stnum = indices_stnum[0]
if len(indices_stnum) != 1:
 print('No matching storm number found or too many matches found. Skipping.')
 sys.exit()
stnum = stnum[indices_stnum]
date = np.array(date)[indices_stnum.astype(int)] # Turn list to array
shr_hdg = shr_hdg[:, indices_stnum]
latc = latc[:, indices_stnum]
lonc = lonc[:, indices_stnum]
#
# Now, get the appropriate lead-time for the shear vector heading
shr_hdg = shr_hdg[lt_stm_idx]
latc = latc[lt_stm_idx]
lonc = lonc[lt_stm_idx]
if (latc == -999.9) or (lonc == -999.9):
 print('Missing clon/clat data. Skipping.')
 sys.exit()
if (shr_hdg == -999.9):
 print('Missing shear data. Skipping.')
 sys.exit()
lonc = lonc - 360.0
#
# Convert shear heading to radians and to an ordinary trigometric angle
th_shr = 90 - shr_hdg
if (th_shr < 0):
 th_shr = 360 + th_shr
th_shr = th_shr * np.pi / 180.0
#
nx = len(lon)
ny = len(lat)
nlon = nx
nlat = ny
lon2d = np.tile(lon, (ny, 1))
lat2d = np.transpose(np.tile(lat, (nx, 1)))
#
# Create Cartesian grid normalized by latitude
# Make 500 x 500 km grid
#
x = 111000.0 * (lon2d - 0.5 * (max(lon) - min(lon)) - min(lon)) * np.cos(lat2d * np.pi / 180.)
y = 111000.0 * (lat2d -  0.5 * (max(lat) - min(lat)) - min(lat))
#
# Get ATCF center
xc_atcf = 111000.0 * (lonc - 0.5 * (max(lon) - min(lon)) - min(lon)) * np.cos(latc * np.pi / 180.)
yc_atcf = 111000.0 * (latc - 0.5 * (max(lat) - min(lat)) - min(lat))
#
# Refine center based on ATCF estimate
x_new_atcf = x - xc_atcf
y_new_atcf = y - yc_atcf
#
radius = np.sqrt(x_new_atcf ** 2 + y_new_atcf ** 2)
index_sector = np.logical_and((radius <= 100000.), (radius >= 0))
# 
# Calculate cylindrical wind components near surface
#
# Calculate vorticity
vort = vortz(u_raw, v_raw, x_new_atcf, y_new_atcf)
vort = ndimage.gaussian_filter(vort, sigma=10, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
#
# Calculate centroid of vorticity
xc_vort = np.sum(x[index_sector] * vort[index_sector]) / np.sum(vort[index_sector])
yc_vort = np.sum(y[index_sector] * vort[index_sector]) / np.sum(vort[index_sector])
#
# Figure out center lat/lon
cenlat = yc_vort / 111000.0 + 0.5 * (max(lat) - min(lat)) + min(lat)
cenlon = xc_vort / (111000.0 * np.cos(cenlat * np.pi / 180.0)) + 0.5 * (max(lon) - min(lon)) + min(lon)
#
# Make new grid for interpolation
x1 = np.arange(-250000, 251000, 4000) 
y1 = np.arange(-250000, 251000, 4000)
[x_reg, y_reg] = np.meshgrid(x1, y1)
#
# Refine center based on centroid of vorticity
x = x - xc_vort
y = y - yc_vort
#
# Rotate now
xp = x * np.cos(th_shr) + y * np.sin(th_shr)
yp = -x * np.sin(th_shr) + y * np.cos(th_shr)
#
# Calculate v_rad and v_tan components of the wind
[u_vort, v_vort] = wc(x, y, u_raw, v_raw)
nz = len(lvls)
u_xy = np.zeros((nz, len(y1), len(x1)))
#
# Interpolation
print('Conducting interpolation of tangential component of wind')
point_x = xp[::inc_skip, ::inc_skip]
point_y = yp[::inc_skip, ::inc_skip]
point_x = point_x.ravel()
point_y = point_y.ravel()
values = v_vort[::inc_skip, ::inc_skip]
values = values.ravel()
points = np.transpose(np.stack((point_x, point_y)))
v_xy_10m = griddata(points, values, (x_reg, y_reg), method = 'linear')
print('Conducting interpolation of the radial component of wind')
values = u_vort[::inc_skip, ::inc_skip]
values = values.ravel()
u_xy_10m = griddata(points, values, (x_reg, y_reg), method = 'linear')
print('Conducting interpolation of theta_e')
values = th_2m[::inc_skip, ::inc_skip]
values = values.ravel()
th_xy_2m = griddata(points, values, (x_reg, y_reg), method = 'linear')
print('Conducting interpolation of tcolc')
values = tcolc[::inc_skip, ::inc_skip]
values = values.ravel()
tcolc_xy = griddata(points, values, (x_reg, y_reg), method = 'linear')
#
u_xy = np.zeros((nz, len(y1), len(x1)))
v_xy = np.zeros((nz, len(y1), len(x1)))
th_xy = np.zeros((nz, len(y1), len(x1)))
for iz in range(0, nz):
 if lvls[iz] > 10000.0:
  (u_rad, v_tan) = (wc(x[::inc_skip, ::inc_skip], y[::inc_skip, ::inc_skip], 
  			np.squeeze(u_p[iz, ::inc_skip, ::inc_skip]), 
			np.squeeze(v_p[iz, ::inc_skip, ::inc_skip])))
  print('Interpolating grids on pressure level ' + str(lvls[iz]))
  values = u_rad.ravel()
  u_xy[iz, :, :] = griddata(points, values, (x_reg, y_reg), method = 'linear')
  values = v_tan.ravel()
  v_xy[iz, :, :] = griddata(points, values, (x_reg, y_reg), method = 'linear')
  values = th_p[iz, ::inc_skip, ::inc_skip]
  values = values.ravel()
  th_xy[iz, :, :] = griddata(points, values, (x_reg, y_reg), method = 'linear')
print("--- %s seconds ---" % (time.time() - start_time))
#
nx = len(x1)
ny = len(y1)
#
n1 = 1
#
# Output data to new netCDF file
f = Dataset(file_out, 'w', format = 'NETCDF4')
# - define a set of dimensions used for variables
lv_ISBL0 = f.createDimension('lv_ISBL0', 3)
lat_0 = f.createDimension('lat_0', nlat)
lon_0 = f.createDimension('lon_0', nlon)
x_0 = f.createDimension('x_0', nx)
y_0 = f.createDimension('y_0', ny)
sclr = f.createDimension('sclr', n1)
# - set up variables
THETA_E_L100 = f.createVariable('THETA_E_L100', np.float32, ('lv_ISBL0', 'y_0', 'x_0'))
THETA_E_L103 = f.createVariable('THETA_E_L103', np.float32, ('y_0', 'x_0'))
TCOLC_L103 = f.createVariable('TCOLC_E_L103', np.float32, ('y_0', 'x_0'))
U_RAD_L100 = f.createVariable('U_RAD_L100', np.float32, ('lv_ISBL0', 'y_0', 'x_0')) 
U_RAD_L103 = f.createVariable('U_RAD_L103', np.float32, ('y_0', 'x_0'))
V_TAN_L100 = f.createVariable('V_TAN_L100', np.float32, ('lv_ISBL0', 'y_0', 'x_0'))
V_TAN_L103 = f.createVariable('V_TAN_L103', np.float32, ('y_0', 'x_0'))
SHR_HDG = f.createVariable('SHR_HDG', np.float32, ('sclr'))
CENLAT = f.createVariable('CENLAT', np.float32, ('sclr'))
CENLON = f.createVariable('CENLON', np.float32, ('sclr'))
y_redo = f.createVariable('y_0', np.float32, ('y_0'))
x_redo = f.createVariable('x_0', np.float32, ('x_0'))
lat_redo = f.createVariable('lat_0', np.float32, ('lat_0'))
lon_redo = f.createVariable('lon_0', np.float32, ('lon_0'))
lvls_redo = f.createVariable('lv_ISBL0', np.float32, ('lv_ISBL0'))
# - variable attributes
THETA_E_L100.initial_time = initial_time
THETA_E_L100.forecast_time_units = 'hours'
THETA_E_L100.forecast_time = forecast_time
THETA_E_L100.level_type = 'Isobaric surface (Pa)'
THETA_E_L100.parameter_category = 'Equivalent Pot. Temp. (K)'
THETA_E_L100.grid_type = 'y/x'
THETA_E_L100.units = 'K'
#
THETA_E_L103.initial_time = initial_time
THETA_E_L103.forecast_time_units = 'hours'
THETA_E_L103.forecast_time = forecast_time
THETA_E_L103.level = 2.0
THETA_E_L013 = 'Specified heigh level above ground (m)'
THETA_E_L103.parameter_category = 'Equivalent Pot. Temp. (K)'
THETA_E_L103.grid_type = 'y/x'
THETA_E_L103.units = 'K'
#
TCOLC_L103.initial_time = initial_time
TCOLC_L103.forecast_time_units = 'hours'
TCOLC_L103.forecast_time = forecast_time
TCOLC_L103.level = 0.0
TCOLC_L103.parameter_category = 'Total column-integrated condensate'
TCOLC_L103.grid_type = 'y/x'
TCOLC_L103.units = 'kg m-2'
#
U_RAD_L100.initial_time = initial_time
U_RAD_L100.forecast_time_units = 'hours'
U_RAD_L100.forecast_time = forecast_time
U_RAD_L100.level_type = 'Isobaric surface (Pa)'
U_RAD_L100.parameter_category = 'Radial wind component (m s-1)'
U_RAD_L100.grid_type = 'y/x'
U_RAD_L100.units = 'm s-1'
#
U_RAD_L103.initial_time = initial_time
U_RAD_L103.forecast_time_units = 'hours'
U_RAD_L103.forecast_time = forecast_time
U_RAD_L103.level = 10.0
U_RAD_L103.parameter_category = 'Radial wind component (m s-1)'
U_RAD_L103.grid_type = 'y/x'
U_RAD_L103.units = 'm s-1'
#
V_TAN_L100.initial_time = initial_time
V_TAN_L100.forecast_time_units = 'hours'
V_TAN_L100.forecast_time = forecast_time
V_TAN_L100.level_type = 'Isobaric surface (Pa)'
V_TAN_L100.parameter_category = 'Tangential wind component (m s-1)'
V_TAN_L100.grid_type = 'y/x'
V_TAN_L100.units = 'm s-1'
#
V_TAN_L103.initial_time = initial_time
V_TAN_L103.forecast_time_units = 'hours'
V_TAN_L103.forecast_time = forecast_time
V_TAN_L103.level = 10.0
V_TAN_L103.parameter_category = 'Tangential wind component (m s-1)'
V_TAN_L103.grid_type = 'y/x'
V_TAN_L103.units = 'm s-1'
#
SHR_HDG.initial_time = initial_time
SHR_HDG.forecast_time_units = 'Degrees'
SHR_HDG.forecast_time = forecast_time
SHR_HDG.parameter_category = 'Shear Heading (degrees north)'
SHR_HDG.grid_type = 'scalar'
SHR_HDG.units = 'Degrees'
#
CENLAT.initial_time = initial_time
CENLAT.forecast_time_units = 'Degrees'
CENLAT.forecast_time = forecast_time
CENLAT.parameter_category = 'Center Latitude (N)'
CENLAT.grid_type = 'scalar'
CENLAT.units = 'Degrees N'
#
CENLON.initial_time = initial_time
CENLON.forecast_time_units = 'Degrees'
CENLON.forecast_time = forecast_time
CENLON.parameter_category = 'Center Longitude (E)'
CENLON.grid_type = 'scalar'
CENLON.units = 'Degrees E'
#
y_redo.units = 'm'
y_redo.grid_type = 'Y'
y_redo.long_name = 'y'
#
x_redo.units = 'm'
x_redo.grid_type = 'X'
x_redo.long_name = 'x'
#
lat_redo.units = 'degrees_north'
lat_redo.grid_type = 'Latitude'
lat_redo.long_name = 'latitude'
#
lon_redo.units = 'degrees_east'
lon_redo.grid_type = 'Longitude'
lon_redo.long_name = 'longitude'
#
lvls_redo.units = 'Pa'
lvls_redo.long_name = 'Isobaric surface'
#
# - fill variables
THETA_E_L100[:] = th_xy
THETA_E_L103[:] = th_xy_2m
TCOLC_L103[:] = tcolc_xy
U_RAD_L100[:] = u_xy
U_RAD_L103[:] = u_xy_10m
V_TAN_L100[:] = v_xy
V_TAN_L103[:] = v_xy_10m
y_redo[:] = y1
x_redo[:] = x1
lat_redo[:] = lat
lon_redo[:] = lon + 360.0
lvls_redo[:] = lvls[0:4]
SHR_HDG[:] = shr_hdg
CENLAT[:] = cenlat
CENLON[:] = cenlon
#
f.close
#
# Sanity check plots
#print('Plotting graphics')
#fig = plt.figure(figsize=(8.45, 7), dpi = 100)
#CS = plt.contourf(x_reg / 1000.0, y_reg / 1000.0, np.squeeze(v_xy_10m))
#plt.axis('equal')
#plt.xlim(-200, 200)
#plt.ylim(-200, 200)
#plt.colorbar()
#plt.show()
# 
print('Program to rotate HWRF fields ending')
print(' ')
