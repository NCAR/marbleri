import xarray as xr
from os.path import exists
from keras.utils import Sequence
import numpy as np
import re


class HWRFStep(object):
    def __init__(self, filename):
        if not exists(filename):
            raise FileNotFoundError(filename + " not found.")
        self.filename = filename
        filename_components = self.filename.split("/")[-1].split(".")
        name_number_split = re.search("[0-9]", filename_components[0]).start()
        self.storm_name = filename_components[0][:name_number_split]
        self.storm_number = filename_components[0][name_number_split:]
        self.run_date = filename_components[1]
        self.forecast_hour = filename_components[2][1:]
        self.ds = xr.open_dataset(filename)
        self.levels = self.ds["lv_ISBL0"].values

    def get_variable(self, variable_name, level=None, subset=None):
        if variable_name not in list(self.ds.variables):
            raise KeyError(variable_name + " not available in " + self.filename)
        if subset is None:
            subset_y = slice(0, self.ds.dims["lat_0"])
            subset_x = slice(0, self.ds.dims["lon_0"])
        else:
            subset_y = slice(subset[0], subset[1])
            subset_x = slice(subset[0], subset[1])
        if level is None:
            if len(self.ds.variable.shape) == 3:
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


class HWRFSequence(Sequence):
    def __init__(self, hwrf_files, best_track_files, batch_size):
        self.hwrf_files = hwrf_files
        self.best_track_files = best_track_files
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.hwrf_files) / self.batch_size)

    def __get_item__(self, idx):
        return



