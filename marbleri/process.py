import numpy as np
import pandas as pd
from .nwp import HWRFStep
from os.path import join, exists
import os
from dask.distributed import as_completed
import xarray as xr
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler_classes = {"StandardScaler": StandardScaler,
                  "MinMaxScaler": MinMaxScaler,
                  "RobustScaler": RobustScaler}

def get_hwrf_filenames(best_track_df, hwrf_path, extension=".nc"):
    """
    Assemble the HWRF file names from the columns of the best track dataframe.

    Args:
        best_track_df: `pandas.DataFrame` containing information about each HWRF time step
        hwrf_path: Path to HWRF files
        extension: type of file containing hwrf data.
    Returns:
        Array of HWRF filenames matching each row in the best track dataframe
    """
    hwrf_filenames = []
    for i in range(best_track_df.shape[0]):
        storm_name = best_track_df.loc[i, "STNAM"]
        storm_number = int(best_track_df.loc[i, "STNUM"])
        basin = best_track_df.loc[i, "BASIN"]
        run_date = best_track_df.loc[i, "DATE"]
        forecast_hour = int(best_track_df.loc[i, "TIME"])
        hwrf_filename = join(hwrf_path,
                             f"{storm_name}{storm_number:02d}{basin}.{run_date}.f{forecast_hour:03d}" + extension)
        hwrf_filenames.append(hwrf_filename)
    return np.array(hwrf_filenames)


def get_hwrf_filenames_diff(best_track_df, hwrf_path, diff=24, extension=".nc"):
    """
    Assemble the HWRF file name pairs for creating time difference fields from the columns of the best track dataframe.

    Args:
        best_track_df: `pandas.DataFrame` containing information about each HWRF time step
        hwrf_path: Path to HWRF files
        extension: type of file containing hwrf data.
    Returns:
        hwrf_file_names_start: HWRF files at the beginning of the intensification period
        hwrf_file_names_end: HWRF files at the end of the intensification period.
    """
    hwrf_filenames_start = []
    hwrf_filenames_end = []
    print(best_track_df.columns)
    for i in range(best_track_df.shape[0]):
        storm_name = best_track_df.loc[i, "STNAM"]
        storm_number = int(best_track_df.loc[i, "STNUM"])
        basin = best_track_df.loc[i, "BASIN"]
        run_date = best_track_df.loc[i, "DATE"]
        forecast_hour = int(best_track_df.loc[i, "TIME"])
        fh_end = forecast_hour + diff
        hwrf_filename_start = join(hwrf_path,
                             f"{storm_name}{storm_number:02d}{basin}.{run_date}.f{forecast_hour:03d}" + extension)
        hwrf_filename_end = join(hwrf_path,
                             f"{storm_name}{storm_number:02d}{basin}.{run_date}.f{fh_end:03d}" + extension)
        hwrf_filenames_start.append(hwrf_filename_start)
        hwrf_filenames_end.append(hwrf_filename_end)
    return np.array(hwrf_filenames_start), np.array(hwrf_filenames_end)


def get_var_levels(input_vars, pressure_levels):
    """
    For each input variable, create a list of input variable, pressure level combinations.

    Args:
        input_vars (list): List of input variables without the _L10* designation (e.g., THETA_E, U_TAN, V_RAD)
        pressure_levels (list): List of integer pressure levels in Pa and surface if surface variable is included.

    Returns:
        input_var_levels
    """
    input_var_levels = []
    for input_var in input_vars:
        for pressure_level in pressure_levels:
            if pressure_level == "surface":
                input_var_levels.append((input_var + "_L103", None))
            else:
                input_var_levels.append((input_var + "_L100", pressure_level))
    return input_var_levels


def process_hwrf_time_differences(best_track_df, variable_levels, time_difference_hours,
                                  subset_indices, hwrf_path, out_path):
    """
    Load HWRF run, calculate the grid cell differences between fields at two forecast hours and save the differences
    to netCDF files.

    Args:
        best_track_df: Dataframe containing valid HWRF time steps to be extracted
        variable_levels: list of tuples containing the variable and the pressure level being extracted.
        subset_indices:
        hwrf_path: Path to HWRF netCDF files
        out_path: Path to where difference netCDF files are saved.

    Returns:

    """

    return


def load_hwrf_data(hwrf_file_list, input_var_levels=None):
    data = []
    step_data = []
    for t, train_file in enumerate(hwrf_file_list):
        step = HWRFStep(train_file)
        for var in input_var_levels:
            step_data.append(step.get_variable(var[0], level=var[1]).values)
        step.close()
        del step
        data.append(np.stack(step_data, axis=-1))
        del step_data[:]
    train_data = np.stack(data, axis=0)
    return train_data


def load_data_diff(hwrf_file_list, input_var_levels=None):
    """
    Load HWRF netCDF files that contain the time differences between fields.

    Args:
        hwrf_file_list: List of HWRF file pairs to be loaded.
        input_var_levels: List of input variables in (variable, level) format.

    Returns:

    """
    data = []
    step_data = []
    for t, train_file in enumerate(hwrf_file_list):
        step_start = HWRFStep(train_file[0])
        step_end = HWRFStep(train_file[1])
        for var in input_var_levels:
            step_data.append(step_end.get_variable(var[0], level=var[1]).values - step_start.get_variable(var[0], level=var[1]).values)
        step_end.close()
        step_start.close()
        del step_start, step_end
        data.append(np.stack(step_data, axis=-1))
        del step_data[:]
    train_data = np.stack(data, axis=0)
    return train_data


def load_hwrf_data_distributed(hwrf_file_list, input_var_levels, client):
    """
    Load data in parallel with dask.

    Args:
        hwrf_file_list: List of hwrf filename pairs
        input_var_levels:
        client: Dask distributed client

    Returns:

    """
    n_workers = len(client.cluster.workers)
    slice_indices = np.linspace(0, len(hwrf_file_list), n_workers + 1).astype(int)
    hwrf_subsets = [hwrf_file_list[slice_index:slice_indices[s + 1]]
                    for s, slice_index in enumerate(slice_indices[:-1])]
    out = client.map(load_data_diff, hwrf_subsets, input_var_levels=input_var_levels)
    all_data = np.vstack(client.gather(out))
    return all_data


def coarsen_hwrf_runs(hwrf_files, variable_levels, window_size, subset_indices, out_path,
                      dask_client):
    """
    Aggregate HWRF data spatially to produce lower resolution fields.

    Args:
        hwrf_files: list of HWRF netCDF files
        variable_levels: Tuples of variable and pressure level or None if surface.
        window_size: size of pooling window (usually a power of 2)
        subset_indices: beginning and ending indices of domain
        out_path: path to output coarse files. An additional directory is created
            for files of a certain window size.
        dask_client: Dask Client object.

    Returns:

    """
    futures = []
    sorted_hwrf_files = pd.Series(sorted(hwrf_files))
    hwrf_run_names = sorted_hwrf_files.str.split("/").str[-1].str.split(".").str[:-2].str.join("_")
    hwrf_run_unique_names = np.unique(hwrf_run_names.values)
    coarse_out_path = join(out_path, f"hwrf_coarse_{window_size:02d}")
    if not exists(coarse_out_path):
        os.makedirs(coarse_out_path)
    for hwrf_run_unique_name in hwrf_run_unique_names:
        hwrf_run_indices = np.where(hwrf_run_names == hwrf_run_unique_name)[0]
        futures.append(dask_client.submit(coarsen_hwrf_run_set, sorted_hwrf_files.values[hwrf_run_indices],
                                                           variable_levels, window_size, subset_indices, coarse_out_path))
    dask_client.gather(futures)
    del futures[:]


def coarsen_hwrf_run_set(hwrf_files, variable_levels, window_size, subset_indices, out_path, pool="mean"):
    """
    This function coarse-grains a set of HWRF files and outputs them as a single larger netCDF file.

    Args:
        hwrf_files: List of hwrf filenames that are being combined into one netCDF file
        variable_levels: List of pairs of variable and pressure levels
        window_size (int): Size of pooling window. Recommend something divisible by 2.
        subset_indices: Tuple of beginning and ending indices for subsetting original hwrf data.
        out_path: Path to output file.
        pool: Whether to use mean, max, median, or min pooling

    """
    subset_start = subset_indices[0]
    subset_end = subset_indices[1]
    subset_width = subset_end - subset_start
    coarse_width = subset_width // window_size
    var_level_strs = get_var_level_strings(variable_levels)
    all_coarse_values = {}
    for var_level_str in var_level_strs:
        all_coarse_values[var_level_str] = np.zeros((len(hwrf_files), coarse_width, coarse_width),
                                                                 dtype=np.float32)
    all_coarse_lons = np.zeros((len(hwrf_files), coarse_width), dtype=np.float32)
    all_coarse_lats = np.zeros((len(hwrf_files), coarse_width), dtype=np.float32)
    hwrf_var_attrs = dict()
    hwrf_file_coords = dict()
    storm_ids = [hwrf_file.split("/")[-1][:-3] for hwrf_file in hwrf_files]
    hwrf_file_coords_keys = ["storm_name", "storm_number", "basin", "run_date", "forecast_hour"]
    for file_attr in hwrf_file_coords_keys:
        hwrf_file_coords[file_attr] = []
    for h, hwrf_file in enumerate(hwrf_files):
        hwrf_data = HWRFStep(hwrf_file)
        lon = hwrf_data.ds["lon_0"][slice(subset_start, subset_end)].values
        lat = hwrf_data.ds["lat_0"][slice(subset_end, subset_start, -1)].values
        all_coarse_lons[h] = coarsen_array(lon, window_size=window_size, pool=pool)
        all_coarse_lats[h] = coarsen_array(lat, window_size=window_size, pool=pool)
        for file_attr in hwrf_file_coords_keys:
            hwrf_file_coords[file_attr].append(getattr(hwrf_data, file_attr))
        for v, var_level in enumerate(variable_levels):
            hwrf_variable = hwrf_data.get_variable(*var_level, subset=subset_indices)
            hwrf_var_attrs[var_level_strs[v]] = hwrf_variable.attrs
            var_values = hwrf_variable.values
            if np.any(var_values.ravel() > 1e7):
                var_values[var_values > 1e7] = np.median(var_values)
            all_coarse_values[var_level_strs[v]][h] = coarsen_array(var_values,
                                                                    window_size=window_size, pool=pool)
        hwrf_data.close()
        del hwrf_data
    hwrf_file_coords["storm_ids"] = storm_ids
    all_coarse_da = {}
    var_coords = {"storm": np.arange(len(hwrf_files)),
                  "y": np.arange(coarse_width),
                  "x": np.arange(coarse_width)}
    nc_encoding = {}
    for var_name, coarse_array in all_coarse_values.items():
        all_coarse_da[var_name] = xr.DataArray(coarse_array, dims=("storm", "y", "x"),
                                               coords=var_coords,
                                               attrs=hwrf_var_attrs[var_name],
                                               name=var_name)
        nc_encoding[var_name] = {"zlib": True, "complevel": 2, "least_significant_digit": 4}
    for coord_name, coord in hwrf_file_coords.items():
        hwrf_file_coords[coord_name] = np.array(coord)
    coarse_hwrf_ds = xr.Dataset(all_coarse_da, coords=hwrf_file_coords)
    out_file = join(out_path, hwrf_files[0].split("/")[-1][:-8] + ".nc")
    print(out_file)
    coarse_hwrf_ds.to_netcdf(out_file,
                             encoding=nc_encoding, mode="w")
    coarse_hwrf_ds.close()
    del coarse_hwrf_ds
    return


def coarsen_array(data, window_size=2, pool="mean"):
    window_steps = np.arange(window_size)
    if len(data.shape) == 2:
        all_coarse_data = np.zeros((data.shape[0] // window_size,
                                    data.shape[1] // window_size, window_size ** 2),
                                   dtype=data.dtype)
        step = 0
        for i in window_steps:
            for j in window_steps:
                all_coarse_data[:, :, step] = data[i::window_size, j::window_size]
                step += 1
    else:
        all_coarse_data = np.zeros((data.shape[0] // window_size, window_size),
                                   dtype=data.dtype)
        step = 0
        for i in window_steps:
            all_coarse_data[:, step] = data[i::window_size]
            step += 1
    if pool == "mean":
        coarse_data = all_coarse_data.mean(axis=-1)
    elif pool == "max":
        coarse_data = all_coarse_data.max(axis=-1)
    elif pool == "min":
        coarse_data = all_coarse_data.min(axis=-1)
    elif pool == "median":
        coarse_data = np.median(all_coarse_data, axis=-1)
    else:
        coarse_data = all_coarse_data.mean(axis=-1)
    return coarse_data


def get_var_level_strings(variable_levels):
    """
    Convert a list of tuples (variable, pressure level) to variable_level strings. If the second part
    of the tuple contains None, then the second part of the string states "surface".

    Args:
        variable_levels: list of (variable, pressure level tuples)

    Returns:
        List of variable_level strings.
    """
    var_level_str = []
    for var_level in variable_levels:
        if var_level[1] is not None:
            var_level_str.append("{0}_{1:1.0f}".format(*var_level))
        else:
            var_level_str.append(var_level[0] + "_surface")
    return var_level_str

def normalize_hwrf_loaded_data(hwrf_field_data, var_levels, scale_format="standard", scale_values=None):
    hwrf_norm_data = np.zeros(hwrf_field_data.shape, dtype=hwrf_field_data.dtype)
    if scale_format == "standard":
        scale_columns = ("mean", "sd")
    else:
        scale_columns = ("min", "max")
    if scale_values is None:
        scale_values = pd.DataFrame(0, index=var_levels, columns=scale_columns,
                                    dtype=hwrf_field_data.dtype)
        if scale_format == "standard":
            for v in range(len(var_levels)):
                scale_values.loc[var_levels[v], "mean"] = hwrf_field_data[..., v].mean()
                scale_values.loc[var_levels[v], "sd"] = hwrf_field_data[..., v].std()
        else:
            for v in range(len(var_levels)):
                scale_values.loc[var_levels[v], "min"] = hwrf_field_data[..., v].min()
                scale_values.loc[var_levels[v], "max"] = hwrf_field_data[..., v].max()
    if scale_format == "standard":
        for v in range(len(var_levels)):
            hwrf_norm_data[..., v] = (hwrf_field_data[..., v]- scale_values.loc[var_levels[v], "mean"]) / scale_values.loc[var_levels[v], "sd"]
    else:
        for v in range(len(var_levels)):
            hwrf_norm_data[..., v] = (hwrf_field_data[..., v] - scale_values.loc[var_levels[v], "min"]) / (scale_values.loc[var_levels[v], "max"] - scale_values.loc[var_levels[v], "min"])
    return hwrf_norm_data, scale_values


def discretize_output(y, y_bins):
    """
    Convert continuous values to discrete bins where 1 is marked if the example falls in that bin and 0 otherwise.

    Args:
        y: array of output values
        y_bins: left side of output bins.

    Returns:
        y_disc: binary array marking bin for each example.
    """
    y_disc = np.zeros((y.shape[0], y_bins.size), dtype=np.float32)
    bin_index = np.maximum(np.searchsorted(y_bins, y) - 1, 0)
    y_disc[np.arange(y.shape[0]), bin_index] = 1
    return y_disc
