from marbleri.nwp import BestTrackNetCDF
from marbleri.process import process_hwrf_run, calculate_hwrf_global_norms, calculate_hwrf_local_norms
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
    print(bt_df)
    print(bt_df.columns)
    print(bt_df.loc[0])
    bt_df.to_csv(join(config["out_path"], "best_track_all.csv"), index_label="Index")
    # calculate derived variables in data frame
    hwrf_variables = config["hwrf_variables"]
    hwrf_levels = config["hwrf_levels"]
    hwrf_variable_levels = []
    for var in hwrf_variables:
        if "L100" in var:
            for level in hwrf_levels:
                hwrf_variable_levels.append((var, float(level)))
        else:
            hwrf_variable_levels.append((var, None))
    for vl in hwrf_variable_levels:
        print(vl)
    if config["process_hwrf"]:
        cluster = LocalCluster(n_workers=config["n_workers"])
        client = Client(cluster)
        hwrf_files = get_hwrf_filenames(bt_df, config["hwrf_path"])
        if config["normalize"] == "local":
            norm_values = calculate_hwrf_local_norms(hwrf_files, hwrf_variable_levels, config["out_path"], client)
            global_norm = False
        else:
            norm_values = calculate_hwrf_global_norms(hwrf_files, hwrf_variable_levels, config["out_path"], client)
            global_norm = True
        hwrf_out = join(config["out_path"], "hwrf")
        if not exists(hwrf_out):
            os.makedirs(hwrf_out)
        # in parallel extract variables from each model run, subset center from rest of grid and save to other
        # netCDF files
        process_all_hwrf_runs(bt_df, hwrf_variable_levels, norm_values, global_norm, config["hwrf_path"],
                              hwrf_out, config["n_workers"], client)
        client.close()
        cluster.close()

    return


if __name__ == "__main__":
    main()
