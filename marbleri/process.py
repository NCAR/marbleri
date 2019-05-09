import numpy as np
import pandas as pd
from .nwp import HWRFStep
from os.path import join
from dask.distributed import as_completed
import xarray as xr


def get_hwrf_filenames(best_track_df, hwrf_path):
    hwrf_filenames = []
    for i in range(best_track_df.shape[0]):
        storm_name = best_track_df.loc[i, "STNAM"]
        storm_number = int(best_track_df.loc[i, "STNUM"])
        basin = best_track_df.loc[i, "BASIN"]
        run_date = best_track_df.loc[i, "DATE"]
        forecast_hour = int(best_track_df.loc[i, "TIME"])
        hwrf_filename = join(hwrf_path, f"{storm_name}{storm_number:02d}{basin}.{run_date}.f{forecast_hour:03d}.nc")
        hwrf_filenames.append(hwrf_filename)
    return hwrf_filenames


def process_all_hwrf_runs(best_track_df, variable_levels, norm_values, global_norm, hwrf_path, out_path,
                          n_workers, dask_client):
    futures = []
    for i in range(np.minimum(n_workers, best_track_df.shape[0])):
        storm_name = best_track_df.loc[i, "STNAM"]
        storm_number = int(best_track_df.loc[i, "STNUM"])
        basin = best_track_df.loc[i, "BASIN"]
        run_date = best_track_df.loc[i, "DATE"]
        forecast_hour = int(best_track_df.loc[i, "TIME"])
        futures.append(dask_client.submit(process_hwrf_run, run_date, storm_name, storm_number, basin, forecast_hour,
                                                           variable_levels, hwrf_path, out_path,
                                                           norm_values, global_norm))
    i = np.minimum(n_workers, best_track_df.shape[0])
    ac = as_completed(futures)
    for future in ac:
        future.result()
        if i < best_track_df.shape[0]:
            storm_name = best_track_df.loc[i, "STNAM"]
            storm_number = int(best_track_df.loc[i, "STNUM"])
            basin = best_track_df.loc[i, "BASIN"]
            run_date = best_track_df.loc[i, "DATE"]
            forecast_hour = int(best_track_df.loc[i, "TIME"])
            ac.add(dask_client.submit(process_hwrf_run, run_date, storm_name, storm_number, basin, forecast_hour,
                                                           variable_levels, hwrf_path, out_path,
                                                           norm_values, global_norm))
            i += 1
    return


def process_hwrf_run(run_date, storm_name, storm_number, basin, forecast_hour, variable_levels,
                     hwrf_path, out_path, norm_values, global_norm=False, subset_indices=None,):
    hwrf_filename = join(hwrf_path, f"{storm_name}{storm_number:02d}{basin}.{run_date}.f{forecast_hour:03d}.nc")
    print(hwrf_filename)
    hwrf_data = HWRFStep(hwrf_filename)
    if subset_indices is None:
        subset_start = 0
        subset_end = 601
    else:
        subset_end = subset_indices[1]
        subset_start = subset_indices[0]
    subset_width = subset_end - subset_start
    all_norm_values = np.zeros((len(variable_levels), subset_width, subset_width))
    var_level_str = get_var_level_strings(variable_levels)
    for v, var_level in enumerate(variable_levels):
        var_values = hwrf_data.get_variable(*var_level, subset=subset_indices).values
        if global_norm:
            var_norm_values = (var_values - norm_values.iloc[v, 4]) / norm_values.iloc[v, 5]
        else:
            var_norm_values = (var_values - norm_values[v, :, :, 4]) / norm_values[v, :, :, 5]
        var_norm_values[np.isnan(var_values)] = 0
        all_norm_values[v] = var_norm_values
    ds = xr.DataArray(all_norm_values, dims=("variable", "lat", "lon"), coords={"variable": var_level_str,
                                                                                "lat": np.arange(subset_width),
                                                                                "lon": np.arange(subset_width)},
                      name="hwrf_norm")
    ds.to_netcdf(join(out_path, f"{storm_name}{storm_number:02d}{basin}.{run_date}.f{forecast_hour:03d}.nc"))
    return 0


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


def calculate_hwrf_global_norms(hwrf_files, variable_levels, out_path, dask_client):
    """
    Calculate the mean and variance for each HWRF input variable across all valid pixel values.

    Args:
        hwrf_files: List of HWRF files to be processed
        variable_levels: List of variable, level tuples (None for level if surface variable)
        out_path: Path where global norm statistics are saved
        dask_client: dask distributed Client object for parallel processing.

    Returns:

    """
    hwrf_futures = []
    global_stats = np.zeros((len(variable_levels), 6))
    for hwrf_filename in hwrf_files:
        hwrf_futures.append(dask_client.submit(hwrf_step_global_sums, hwrf_filename, variable_levels))
    for hwrf_future in as_completed(hwrf_futures):
        global_stats[:, 0:2] += hwrf_future.result()
    del hwrf_futures[:]
    global_stats[:, 4] = global_stats[:, 0] / global_stats[:, 1]
    for hwrf_filename in hwrf_files:
        hwrf_futures.append(dask_client.submit(hwrf_step_global_variances, hwrf_filename,
                                               variable_levels, global_stats[:, 4]))
    for hwrf_future in as_completed(hwrf_futures):
        global_stats[:, 2:4] += hwrf_future.result()
    del hwrf_futures[:]
    global_stats[:, 5] = np.sqrt(global_stats[:, 2] / (global_stats[:, 3] - 1.0))
    var_level_str = get_var_level_strings(variable_levels)
    df_columns = ["sum", "sum_count", "sum_squared_diff", "sum_squared_diff_count", "mean", "standard_dev"]
    global_stats_df = pd.DataFrame(global_stats, index=var_level_str, columns=df_columns)
    global_stats_df.to_csv(join(out_path, "hwrf_global_norm_stats.csv"), index_label="variable_level")
    return global_stats_df


def hwrf_step_global_sums(hwrf_filename, variable_levels):
    hwrf_data = HWRFStep(hwrf_filename)
    sum_counts = np.zeros((len(variable_levels), 2))
    for v, var_level in enumerate(variable_levels):
        sum_counts[v, 0] = np.nansum(hwrf_data.get_variable(var_level[0], var_level[1]))
        sum_counts[v, 1] = np.count_nonzero(~np.isnan(hwrf_data.get_variable(var_level[0], var_level[1])))
    hwrf_data.close()
    return sum_counts


def hwrf_step_global_variances(hwrf_filename, variable_levels, variable_means):
    hwrf_data = HWRFStep(hwrf_filename)
    sum_counts = np.zeros((len(variable_levels), 2))
    for v, var_level in enumerate(variable_levels):
        sum_counts[v, 0] = np.nansum((hwrf_data.get_variable(var_level[0], var_level[1]) - variable_means[v]) ** 2)
        sum_counts[v, 1] = np.count_nonzero(~np.isnan(hwrf_data.get_variable(var_level[0], var_level[1])))
    hwrf_data.close()
    return sum_counts


def calculate_hwrf_local_norms(hwrf_files, variable_levels, out_path, dask_client):
    hwrf_futures = []
    local_stats = np.zeros((len(variable_levels), 601, 601, 6))
    for hwrf_filename in hwrf_files:
        hwrf_futures.append(dask_client.submit(hwrf_step_local_sums, hwrf_filename, variable_levels))
    for hwrf_future in as_completed(hwrf_futures):
        local_stats[:, :, :, 0:2] += hwrf_future.result()
    del hwrf_futures[:]
    local_stats[:, :, :, 4] = local_stats[:, :, :, 0] / local_stats[:, :, :, 1]
    for hwrf_filename in hwrf_files:
        hwrf_futures.append(dask_client.submit(hwrf_step_local_variances, hwrf_filename,
                                               variable_levels, local_stats[:, 4]))
    for hwrf_future in as_completed(hwrf_futures):
        local_stats[:, :, :, 2:4] += hwrf_future.result()
    local_stats[:, :, :, 5] = np.sqrt(local_stats[:, :, :, 2] / (local_stats[:, :, :, 3] - 1.0))
    var_level_str = get_var_level_strings(variable_levels)
    df_columns = ["sum", "sum_count", "sum_squared_diff", "sum_squared_diff_count", "mean", "standard_dev"]
    local_stat_data = xr.DataArray(local_stats, dims=("variable", "lat", "lon", "statistic"),
                 coords={"variable": var_level_str, "lat": np.arange(601),
                         "lon": np.arange(601), "statistic": df_columns}, name="local_norm_stats")
    local_stat_data.to_netcdf(join(out_path, "hwrf_local_norm_stats.nc"))
    return local_stat_data


def hwrf_step_local_sums(hwrf_filename, variable_levels):
    hwrf_data = HWRFStep(hwrf_filename)
    sum_counts = np.zeros((len(variable_levels), 601, 601, 2))
    for v, var_level in enumerate(variable_levels):
        var_data = hwrf_data.get_variable(var_level[0], var_level[1]).values
        sum_counts[v, :, :, 0] = np.where(~np.isnan(var_data), var_data, 0)
        sum_counts[v, :, :, 1] = np.where(~np.isnan(var_data), 1, 0)
    hwrf_data.close()
    return sum_counts


def hwrf_step_local_variances(hwrf_filename, variable_levels, variable_means):
    hwrf_data = HWRFStep(hwrf_filename)
    sum_counts = np.zeros((len(variable_levels), 601, 601, 2))
    for v, var_level in enumerate(variable_levels):
        var_data = hwrf_data.get_variable(var_level[0], var_level[1]).values
        sum_counts[v, 0] = np.where(~np.isnan(var_data), (var_data - variable_means[v]) ** 2, 0)
        sum_counts[v, 1] = np.where(~np.isnan(var_data), 1, 0)
    hwrf_data.close()
    return sum_counts
