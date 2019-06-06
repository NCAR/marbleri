from marbleri.nwp import BestTrackNetCDF
from marbleri.process import calculate_hwrf_global_norms, calculate_hwrf_local_norms
from marbleri.process import process_all_hwrf_runs, get_hwrf_filenames
import argparse
from os.path import join, exists
import yaml
from dask.distributed import LocalCluster, Client
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config filepath")
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
    bt_df = bt_nc.to_dataframe(best_track_variables, dropna=True)
    bt_df.to_csv(join(config["out_path"], "best_track_all.csv"), index_label="Index")
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
    if config["process_hwrf"]:
        print(config["n_workers"])
        print(config["dask_worker"])
        cluster = LocalCluster(n_workers=0, threads_per_worker=1)
        for i in range(config["n_workers"]):
            cluster.start_worker(**config["dask_worker"])
        client = Client(cluster)
        print(client)
        print(cluster)
        if config["normalize"] == "local":
            print("local normalization")
            norm_values = calculate_hwrf_local_norms(hwrf_files, hwrf_variable_levels, subset_indices,
                                                     config["out_path"], client, config["n_workers"])
            global_norm = False
        else:
            print("global normalization")
            norm_values = calculate_hwrf_global_norms(hwrf_files, hwrf_variable_levels, config["out_path"], client)
            global_norm = True
        hwrf_out_path = config["out_path"]
        hwrf_out_file = config[""]
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
