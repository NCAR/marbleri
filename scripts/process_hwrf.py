from marbleri.nwp import BestTrackNetCDF
import argparse
from os.path import join, exists
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config filepath")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError("Config file {0} not found.".format(args.config))
    with open(args.config, "rb") as config_file:
        config = yaml.load(config_file, yaml.Loader)
    best_track_path = config["best_track_path"]
    best_track_variables = config["best_track_variables"]
    # load best track data
    bt_nc = BestTrackNetCDF(file_path=best_track_path)
    # convert best track data to data frame and filter out NaNs in HWRF and best track winds
    bt_df = bt_nc.to_dataframe(best_track_variables, dropna=True)
    print(bt_df)
    print(bt_df.columns)
    print(bt_df.loc[0])
    # calculate derived variables in data frame

    # in parallel extract variables from each model run, subset center from rest of grid and save to other
    # netCDF files
    return


if __name__ == "__main__":
    main()
