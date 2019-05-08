import xarray as xr
from keras.utils import Sequence
from os.path import exists, join
import pandas as pd
import numpy as np
import re


class HWRFStep(object):
    """
    HWRFStep loads a single HWRF netCDF file and enables the extraction of variables in the file.

    Attributes:
        filename: Full path and name of the HWRF file
        decode_cf: Whether to decode file coordinate variables based on the CF conventions in xarray.open_dataset.
        decode_times: Whether or not to convert the time variables from their raw formats to datetime.
        decode_coords: Whether or not to calculate the latitude and longitude grids with xarray.
        storm_name: Name of the storm extracted from the filename.
        storm_number: Number associated with the storm.
        basin: Which ocean basin the storm occurs in. "e" is for East Pacific and "l" is for Atlantic.
        run_date: The date that HWRF was initialized
        forecast_hour: How many hours since the run date. This is when the model output is valid.
        ds: `xarray.Dataset` object containing the model variables.
        levels: Pressure levels for the 3D variables stored in the file. The pressures are in Pa.
    """
    def __init__(self, filename, decode_cf=False, decode_times=False, decode_coords=False):
        if not exists(filename):
            raise FileNotFoundError(filename + " not found.")
        self.filename = filename
        filename_components = self.filename.split("/")[-1].split(".")
        name_number_split = re.search("[0-9]", filename_components[0]).start()
        self.storm_name = filename_components[0][:name_number_split]
        self.storm_number = filename_components[0][name_number_split:].replace("l", "")
        if "e" in self.storm_number:
            self.basin = "e"
            self.storm_number = self.storm_number[:-1]
        else:
            self.basin = "l"
        self.run_date = filename_components[1]
        self.forecast_hour = int(filename_components[2][1:])
        self.ds = xr.open_dataset(filename, decode_cf=decode_cf, decode_times=decode_times, decode_coords=decode_coords)
        self.levels = self.ds["lv_ISBL0"].values

    def get_variable(self, variable_name, level=None, subset=None):
        """
        Extract a particular variable from the file

        Args:
            variable_name:
            level:
            subset:

        Returns:

        """
        if variable_name not in list(self.ds.variables):
            raise KeyError(variable_name + " not available in " + self.filename)
        if subset is None:
            subset_y = slice(0, self.ds.dims["lat_0"])
            subset_x = slice(0, self.ds.dims["lon_0"])
        else:
            subset_y = slice(subset[0], subset[1])
            subset_x = slice(subset[0], subset[1])
        if level is None:
            if len(self.ds.variables[variable_name].shape) == 3:
                level_index = 0
            else:
                level_index = -1
        else:
            if level in self.levels:
                level_index = np.where(self.levels == level)[0][0]
            else:
                raise KeyError("level {0} not found.".format(level))
        if level_index >= 0:
            var_data = self.ds[variable_name][level_index, subset_y, subset_x]
        else:
            var_data = self.ds[variable_name][subset_y, subset_x]
        return var_data

    def close(self):
        self.ds.close()
        self.ds = None


class BestTrackNetCDF(object):
    def __init__(self,
                 atl_filename="diag_2015_2017_adecks_atl_bug_corrected.nc",
                 epo_filename="diag_2015_2017_adecks_epo_bug_corrected.nc",
                 file_path="/glade/p/ral/nsap/rozoff/hfip/besttrack_predictors/"):
        self.atl_filename = atl_filename
        self.epo_filename = epo_filename
        self.file_path = file_path
        self.bt_ds = dict()
        self.bt_ds["l"] = xr.open_dataset(join(self.file_path, self.atl_filename))
        self.bt_ds["e"] = xr.open_dataset(join(self.file_path, self.epo_filename))
        self.bt_runs = dict()
        self.run_columns = ["DATE", "STNAM", "STNUM", "BASIN"]
        for basin in ["l", "e"]:
            self.bt_ds[basin]["vmax_bt_newer"] = xr.DataArray(self.bt_ds[basin]["vmax_bt_new"], dims=("time", "run"))
        for basin in self.bt_ds.keys():
            self.bt_runs[basin] = self.bt_ds[basin][self.run_columns].to_dataframe()
            for col in self.bt_runs[basin].columns:
                self.bt_runs[basin][col] = self.bt_runs[basin][col].str.strip().str.decode("utf-8")

    def get_storm_variables(self, variables, run_date, storm_name, storm_number, basin, forecast_hour):
        b_runs = self.bt_runs[basin]
        run_index = np.where((b_runs["DATE"] == run_date) &
                             (b_runs["STNAM"] == storm_name) &
                             (b_runs["STNUM"] == storm_number))[0]
        fh_index = np.where(self.bt_ds[basin]["TIME"] == forecast_hour)[0]
        
        bt_values = np.zeros((1, len(variables)))
        if len(run_index) > 0 and len(fh_index) > 0: 
            for v, variable in enumerate(variables):
                bt_values[0, v] = self.bt_ds[basin][variable][fh_index[0], run_index[0]].values
        bt_values[np.isnan(bt_values)] = 0
        return bt_values

    def to_dataframe(self, variables, dropna=True):
        basin_dfs = []
        for basin in self.bt_ds.keys():
            print(basin)
            basin_dfs.append(pd.merge(self.bt_runs[basin], self.bt_ds[basin][variables].to_dataframe(), how="right",
                                      left_index=True, right_index=True))
            print(basin_dfs[-1])
            if dropna:
                basin_dfs[-1] = basin_dfs[-1].dropna()
        return pd.concat(basin_dfs, ignore_index=True)

    def close(self):
        for basin in self.bt_ds.keys():
            self.bt_ds[basin].close()
            del self.bt_ds[basin]




class HWRFSequence(Sequence):
    """


    """
    def __init__(self, hwrf_files, best_track, hwrf_variables, hwrf_variable_levels,
                 best_track_input_variables, best_track_label_variables, batch_size,
                 x_start=0, x_end=601, shuffle=True):
        self.hwrf_files = hwrf_files
        self.best_track = best_track
        self.hwrf_variables = hwrf_variables
        self.hwrf_variable_levels = hwrf_variable_levels
        self.num_hwrf_variables = len(self.hwrf_variables)
        self.best_track_input_variables = best_track_input_variables
        self.num_best_track_input_variables = len(best_track_input_variables)
        self.best_track_label_variables = best_track_label_variables
        self.num_best_track_label_variables = len(best_track_label_variables)
        self.batch_size = batch_size
        self.x_subset = (x_start, x_end)
        self.x_size = x_end - x_start
        self.shuffle = True
        self.indexes = np.arange(len(self.hwrf_files))
        self.on_epoch_end()
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.hwrf_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.hwrf_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_list_temp):
        all_hwrf_patches = np.zeros((index_list_temp.size, self.x_size, self.x_size, self.num_hwrf_variables), dtype=np.float32)
        all_best_track_input_data = np.zeros((index_list_temp.size, self.num_best_track_input_variables), dtype=np.float32)
        all_best_track_label_data = np.zeros((index_list_temp.size, self.num_best_track_label_variables), dtype=np.float32)
        for i, idx in enumerate(index_list_temp):
            hwrf_file = self.hwrf_files[idx]
            hwrf_data = HWRFStep(hwrf_file)
            for v, variable in enumerate(self.hwrf_variables):
                all_hwrf_patches[i, :, :, v] = hwrf_data.get_variable(variable, level=self.hwrf_variable_levels[v],
                                                            subset=self.x_subset)
                all_hwrf_patches[i, :, :, v][np.isnan(all_hwrf_patches[i, :, :, v])] = np.nanmin(all_hwrf_patches[i, :, :, v])
            hwrf_data.close()
            all_best_track_input_data[i] = self.best_track.get_storm_variables(self.best_track_input_variables,
                                            hwrf_data.run_date, hwrf_data.storm_name, hwrf_data.storm_number,
                                            hwrf_data.basin, hwrf_data.forecast_hour)
            all_best_track_label_data[i] = self.best_track.get_storm_variables(self.best_track_label_variables,
                                                                    hwrf_data.run_date, hwrf_data.storm_name,
                                                                    hwrf_data.storm_number,
                                                                    hwrf_data.basin, hwrf_data.forecast_hour)
        return [all_hwrf_patches, all_best_track_input_data], all_best_track_label_data

