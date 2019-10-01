from marbleri.nwp import BestTrackNetCDF
from marbleri.process import calculate_hwrf_global_norms, calculate_hwrf_local_norms
from marbleri.process import process_all_hwrf_runs, get_hwrf_filenames, coarsen_hwrf_runs
import argparse
from os.path import join, exists
import xarray as xr
import yaml
from dask.distributed import LocalCluster, Client
import os
import numpy as np
import logging
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config filepath")
    parser.add_argument("-c", "--coarse", action="store_true", help="Coarsen HWRF fields.")
    parser.add_argument("-s", "--stat", action="store_true", help="Calculate mean and standard deviation for grids")
    parser.add_argument("-n", "--norm", action="store_true", help="Normalize gridded HWRF fields.")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError("Config file {0} not found.".format(args.config))
    with open(args.config, "rb") as config_file:
        config = yaml.load(config_file, yaml.Loader)
    best_track_path = config["best_track_path"]
    out_path = config["out_path"]
    if not exists(out_path):
        os.makedirs(out_path)
    best_track_variables = config["best_track_variables"]
    # load best track data
    logging.info("Processing Best Track Data")
    bt_nc = BestTrackNetCDF(file_path=best_track_path)
    # convert best track data to data frame and filter out NaNs in HWRF and best track winds
    #bt_df = bt_nc.to_dataframe(best_track_variables, dropna=False)
    #for col in bt_df.columns[4:]:
    #    print(col, np.count_nonzero(np.isnan(bt_df[col].values)))
    bt_df = bt_nc.to_dataframe(best_track_variables, dropna=True)
    
    logging.info(f"BT Shape {bt_df.shape[0]:d}")
    bt_df.to_csv(join(out_path, "best_track_all.csv"), index_label="Index")
    if config["process_hwrf"] and (args.coarse or args.stat or args.norm): 
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
        logging.info("Starting Dask Cluster")
        cluster = LocalCluster(n_workers=0, threads_per_worker=1)
        for i in range(config["n_workers"]):
            cluster.start_worker(**config["dask_worker"])
        client = Client(cluster)
        norm_values = None
        global_norm = False
        if args.coarse:
            coarsen_hwrf_runs(hwrf_files, hwrf_variable_levels, config["window_size"],
                              subset_indices, out_path, client)
        if args.stat:
            if config["normalize"] == "local":
                logging.info("local normalization")
                norm_values = calculate_hwrf_local_norms(hwrf_files, hwrf_variable_levels, subset_indices,
                                                         out_path, client, config["n_workers"])
                global_norm = False
            else:
                logging.info("global normalization")
                norm_values = calculate_hwrf_global_norms(hwrf_files, hwrf_variable_levels, out_path, client)
                global_norm = True
        if args.norm:
            if norm_values is None:
                norm_ds = xr.open_dataset(join(out_path, "hwrf_local_norm_stats.nc"))
                norm_values = norm_ds["local_norm_stats"].values
                norm_ds.close()
            hwrf_out_path = join(out_path, "hwrf_norm")
            if not exists(hwrf_out_path):
                os.makedirs(hwrf_out_path)
            # in parallel extract variables from each model run, subset center from rest of grid and save to other
            # netCDF files
            logging.info("process HWRF runs")
            process_all_hwrf_runs(hwrf_files, hwrf_variable_levels, subset_indices, norm_values, global_norm,
                              hwrf_out_path, config["n_workers"], client)
        client.close()
        cluster.close()

    return


if __name__ == "__main__":
    main()
