from marbleri.nwp import BestTrackNetCDF
from marbleri.process import calculate_hwrf_global_norms, calculate_hwrf_local_norms
from marbleri.process import process_all_hwrf_runs, get_hwrf_filenames
import argparse
from os.path import join, exists
import xarray as xr
import yaml
from dask.distributed import LocalCluster, Client
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config filepath")
    parser.add_argument("-s", "--stat", action="store_true", help="Calculate mean and standard deviation for grids")
    parser.add_argument("-n", "--norm", action="store_true", help="Normalize gridded HWRF fields.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError("Config file {0} not found.".format(args.config))
    with open(args.config, "rb") as config_file:
        config = yaml.load(config_file, yaml.Loader)
    best_track_path = config["best_track_path"]
    if not exists(config["out_path"]):
        os.makedirs(config["out_path"])
    best_track_variables = config["best_track_variables"]
    # load best track data
    bt_nc = BestTrackNetCDF(file_path=best_track_path)
    # convert best track data to data frame and filter out NaNs in HWRF and best track winds
    bt_df = bt_nc.to_dataframe(best_track_variables, dropna=False)
    for col in bt_df.columns[4:]:
        print(col, np.count_nonzero(np.isnan(bt_df[col].values)))
    bt_df = bt_nc.to_dataframe(best_track_variables, dropna=True)
    
    print("BT Shape", bt_df.shape)
    bt_df.to_csv(join(config["out_path"], "best_track_all.csv"), index_label="Index")
    if config["process_hwrf"] and (args.stat or args.norm): 
        # calculate derived variables in data frame
        hwrf_variables = config["hwrf_variables"]
        hwrf_levels = config["hwrf_levels"]
        subset_indices = tuple(config["subset_indices"])
        assert len(subset_indices) == 2
        hwrf_variable_levels = []
        for var in hwrf_variables:
            if "L100" in var:
                for level in hwrf_levels:
                    hwrf_variable_levels.append((var, float(level)))
            else:
                hwrf_variable_levels.append((var, None))
        for vl in hwrf_variable_levels:
            print(vl)
        hwrf_files = get_hwrf_filenames(bt_df, config["hwrf_path"])
        print(config["n_workers"])
        print(config["dask_worker"])
        cluster = LocalCluster(n_workers=0, threads_per_worker=1)
        for i in range(config["n_workers"]):
            cluster.start_worker(**config["dask_worker"])
        client = Client(cluster)
        print(client)
        print(cluster)
        norm_values = None
        global_norm = False
        if args.stat:
            if config["normalize"] == "local":
                print("local normalization")
                norm_values = calculate_hwrf_local_norms(hwrf_files, hwrf_variable_levels, subset_indices,
                                                     config["out_path"], client, config["n_workers"])
                global_norm = False
            else:
                print("global normalization")
                norm_values = calculate_hwrf_global_norms(hwrf_files, hwrf_variable_levels, config["out_path"], client)
                global_norm = True
        if args.norm:
            if norm_values is None:
                norm_ds = xr.open_dataset(join(config["out_path"], "hwrf_local_norm_stats.nc"))
                norm_values = norm_ds["local_norm_stats"].values
                norm_ds.close()
            hwrf_out_path = join(config["out_path"], "hwrf_norm")
            hwrf_out_file = config["out_file"]
            print(hwrf_out_path)
            print(hwrf_out_file)
            if not exists(hwrf_out_path):
                os.makedirs(hwrf_out_path)
            # in parallel extract variables from each model run, subset center from rest of grid and save to other
            # netCDF files
            print("process HWRF runs")
            process_all_hwrf_runs(hwrf_files, hwrf_variable_levels, subset_indices, norm_values, global_norm,
                              hwrf_out_path, hwrf_out_file, config["n_workers"], client)
        client.close()
        cluster.close()

    return


if __name__ == "__main__":
    main()
