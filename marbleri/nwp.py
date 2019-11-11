import xarray as xr
from os.path import exists, join
import pandas as pd
import numpy as np
import re
from glob import glob


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
            subset_y = slice(self.ds.dims["lat_0"], 0, -1)
            subset_x = slice(0, self.ds.dims["lon_0"])
        else:
            subset_y = slice(subset[1], subset[0], -1)
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
    """
    Reads and processes the Best Track NetCDF files.

    """
    def __init__(self,
                 file_path="/glade/p/ral/nsap/rozoff/hfip/besttrack_predictors/",
                 file_start="diag_2015_2017"):
        best_track_files = sorted(glob(join(file_path, file_start + "*.nc")))
        if len(best_track_files) < 2:
            raise FileNotFoundError("Matching best track files not found in " + file_path + "with start " + file_start)
        self.atl_filename = best_track_files[0]
        self.epo_filename = best_track_files[1]
        self.file_path = file_path
        self.bt_ds = dict()
        self.bt_ds["l"] = xr.open_dataset(join(self.file_path, self.atl_filename))
        self.bt_ds["e"] = xr.open_dataset(join(self.file_path, self.epo_filename))
        self.bt_runs = dict()
        self.run_columns = ["DATE", "STNAM", "STNUM", "BASIN"]
        # Some of the variables have (time, nrun) as the dimensions, which causes problems when trying to use
        # the xarray.to_dataframe() function. Changing the dimension from nrun to run fixes the problem.
        for basin in ["l", "e"]:
            for variable in self.bt_ds[basin].variables.keys():
                if self.bt_ds[basin][variable].dims == ("time", "nrun"):
                    self.bt_ds[basin][variable] = xr.DataArray(self.bt_ds[basin][variable], dims=("time", "run"))
        for basin in self.bt_ds.keys():
            self.bt_runs[basin] = self.bt_ds[basin][self.run_columns].to_dataframe()
            for col in self.bt_runs[basin].columns:
                self.bt_runs[basin][col] = self.bt_runs[basin][col].str.decode("utf-8").str.strip()

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

