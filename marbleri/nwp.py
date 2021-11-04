import xarray as xr
from os.path import exists, join
import pandas as pd
import numpy as np
import re
from glob import glob
from pvlib.solarposition import get_solarposition


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
        basin: Which ocean basin the storm occurs in. "e" is for East Pacific, "l" is for Atlantic, "w" is for West
            Pacific, "p" is for South Pacific, and "s" is for Southwest Indian Ocean.
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
        self.storm_number = filename_components[0][name_number_split:-1]
        self.basin = filename_components[0][-1]
        self.run_date_str = filename_components[1]
        self.run_date = pd.Timestamp(self.run_date_str + "00")
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
            subset_y = slice(self.ds.dims["y_0"], 0, -1)
            subset_x = slice(0, self.ds.dims["x_0"])
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

    Attributes:
        file_path: Path to all best track files
        file_start: Common starting name for all best track netCDF files.
        start_date: Initial date of period from which storms are loaded by run date
        end_date: Last date of period from which storms are loaded by run date
        bt_ds: Best Track Dataset object. If multiple files loaded, then open_mfdataset is used.
        bt_runs: Dataframe of the individual HWRF runs.

    """

    def __init__(self,
                 file_path="/glade/p/ral/nsap/rozoff/hfip/besttrack_predictors/",
                 file_start="diag_2015_2017",
                 start_date="2015-03-01", end_date="2016-02-28"):
        self.best_track_files = sorted(glob(join(file_path, file_start + "*.nc")))
        if len(self.best_track_files) == 0:
            raise FileNotFoundError("Matching best track files not found in " + file_path + "with start " + file_start)
        self.file_path = file_path
        self.file_start = file_start
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        if len(self.best_track_files) == 1:
            self.bt_ds = xr.open_dataset(self.best_track_files[0])
        else:
            bt_ds_list = [xr.open_dataset(ds) for ds in self.best_track_files]
            for bt_ds in bt_ds_list:
                for variable in bt_ds.variables.keys():
                    if bt_ds[variable].dims == ("time", "nrun"):
                        bt_ds[variable] = xr.DataArray(bt_ds[variable], dims=("time", "run"))
            self.bt_ds = xr.concat(bt_ds_list, "run")
        self.bt_runs = None
        self.run_columns = ["DATE", "STNAM", "STNUM", "BASIN"]
        self.meta_columns = ["INIT_HOUR", "VALID", "TIME", "LON", "LAT", "STM_SPD", "STM_HDG", "LAND", "VMAX"]
        # Some of the variables have (time, nrun) as the dimensions, which causes problems when trying to use
        # the xarray.to_dataframe() function. Changing the dimension from nrun to run fixes the problem.
        for variable in self.bt_ds.variables.keys():
            if self.bt_ds[variable].dims == ("time", "nrun"):
                self.bt_ds[variable] = xr.DataArray(self.bt_ds[variable], dims=("time", "run"))
        self.bt_runs = self.bt_ds[self.run_columns].to_dataframe()
        for col in self.bt_runs.columns:
            self.bt_runs[col] = self.bt_runs[col].str.decode("utf-8").str.strip()
        run_dates = pd.DatetimeIndex(self.bt_runs["DATE"] + "00")
        run_indices = np.where((run_dates >= self.start_date) & (run_dates <= self.end_date))[0]
        self.bt_ds = self.bt_ds.sel(run=run_indices)
        self.bt_ds["INIT_HOUR"] = self.bt_ds["DATE"].str[-2:].astype(int)
        self.valid_times()
        print(self.bt_ds.run)
        self.bt_runs = self.bt_runs.iloc[run_indices].reset_index(drop=True)

    def valid_times(self, valid_time_name="VALID"):
        run_dates = pd.DatetimeIndex(self.bt_ds["DATE"].to_series().str.decode("utf-8") + "00", tz="UTC")
        forecast_hours = pd.TimedeltaIndex(self.bt_ds["TIME"], unit="hours")
        valid_dates = run_dates.values.reshape(-1, 1) + forecast_hours.values.reshape(1, -1)
        self.bt_ds[valid_time_name] = xr.DataArray(valid_dates.T, dims=("time", "run"), name=valid_time_name)
        self.zenith()
        return

    def get_storm_variables(self, variables, run_date, storm_name, storm_number, basin, forecast_hour):
        b_runs = self.bt_runs
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

    def calc_time_differences(self, variables, time_difference_hours):
        step_diff = int(time_difference_hours // 3)
        for variable in variables:
            if variable not in list(self.bt_ds.variables.keys()):
                raise IndexError(variable + " not found in best track data")
            diff_var = np.ones(self.bt_ds[variable].shape, dtype=np.float32) * np.nan
            diff_var[step_diff:] = self.bt_ds[variable][step_diff:].values - \
                self.bt_ds[variable][:-step_diff].values
            diff_var_name = variable + "_dt_{0:02d}".format(time_difference_hours)
            self.bt_ds[diff_var_name] = xr.DataArray(diff_var, dims=("time", "run"), name=diff_var_name)
        return

    def zenith(self):
        "Calculate the solar zenith angle for each time and location."
        solar_pos_data = get_solarposition(self.bt_ds["VALID"].values.ravel(),
                                           self.bt_ds["LAT"].values.ravel(),
                                           self.bt_ds["LON"].values.ravel())
        self.bt_ds["ZENITH"] = xr.DataArray(solar_pos_data["zenith"].values.reshape(self.bt_ds.dims["time"],
                                                                                    self.bt_ds.dims["run"]),
                                            dims=("time", "run"), name="ZENITH")
        return

    def to_dataframe(self, variables, dropna=True):
        """Convert xarray dataset to pandas DataFrame and filter out forecast hours after storm dies."""
        variables_list = list(variables)
        for m in self.meta_columns:
            if m in variables_list:
                variables_list.remove(m)
        variables_list = self.meta_columns + variables_list

        bt_df = pd.merge(self.bt_runs, self.bt_ds[variables_list].to_dataframe(), how="right",
                            left_index=True, right_on="run", right_index=False)
        if dropna:
            bt_df = bt_df.dropna()
        return bt_df

    def close(self):
        self.bt_ds.close()
        del self.bt_ds
