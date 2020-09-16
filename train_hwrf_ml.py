import argparse
import yaml
from os.path import exists, join
import tensorflow as tf
from marbleri.process import get_hwrf_filenames_diff, get_var_levels, load_hwrf_data_distributed, \
    normalize_hwrf_loaded_data, discretize_output, scaler_classes
from marbleri.models import all_models
from marbleri.nwp import BestTrackNetCDF
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    parser.add_argument("-t", "--train", action="store_true", help="Run neural network training.")
    parser.add_argument("-i", "--interp", action="store_true", help="Run interpretation.")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot interpretation results.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    for k, v in config.items():
        print(k, ":", v)
    hwrf_data_paths = config["hwrf_data_paths"]
    # Initialize GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    data_modes = config["data_modes"]
    conv_subset = config["conv_inputs"]["subset"]
    best_track_nc = {}
    best_track_df = {}
    dt = config["time_difference_hours"]
    best_track_inputs_dt = [f"{bt}_dt_{dt:d}" for bt in config["best_track_inputs"]]
    best_track_input_norm = {}
    input_var_levels = get_var_levels(config["conv_inputs"]["variables"], config["conv_inputs"]["levels"])
    output_field = config["best_track_output"] + f"_dt_{dt:d}"
    hwrf_field_data = {}
    hwrf_norm_data = {}
    output_bins = np.arange(config["output_bins"][0],
                            config["output_bins"][1] + config["output_bins"][2],
                            config["output_bins"][2])
    best_track_output_discrete = {}
    best_track_scaler = scaler_classes[config["best_track_scaler"]]()
    cluster = LocalCluster(n_workers=config["n_workers"], threads_per_worker=1)
    client = Client(cluster)
    conv_scale_values = None
    for mode in data_modes:
        print("Loading " + mode)
        best_track_nc[mode] = BestTrackNetCDF(**config["best_track_data_paths"][mode])
        best_track_nc[mode].calc_time_differences(config["best_track_inputs"], config["time_difference_hours"])
        best_track_nc[mode].calc_time_differences([config["best_track_output"]], config["time_difference_hours"])
        best_track_df[mode] = best_track_nc[mode].to_dataframe(best_track_inputs_dt + [output_field])
        if mode == "train":
            best_track_input_norm[mode] = pd.DataFrame(best_track_scaler.fit_transform(
                best_track_df[mode][best_track_inputs_dt]), columns=best_track_inputs_dt)
        else:
            best_track_input_norm[mode] = pd.DataFrame(best_track_scaler.transform(
                best_track_df[mode][best_track_inputs_dt]), columns=best_track_inputs_dt)
        best_track_output_discrete[mode] = discretize_output(best_track_df[mode][output_field].values, output_bins)
        print(best_track_df[mode])
        hwrf_filenames_start, hwrf_filenames_end = get_hwrf_filenames_diff(best_track_df[mode],
                                                                           hwrf_data_paths[mode],
                                                                           diff=config["time_difference_hours"])
        print(hwrf_filenames_start[0], hwrf_filenames_end[0])
        hwrf_files_se = np.vstack([hwrf_filenames_start, hwrf_filenames_end]).T
        hwrf_field_data[mode] = load_hwrf_data_distributed(hwrf_files_se, input_var_levels, client, subset=conv_subset)
        print("Normalize " + mode)
        hwrf_norm_data[mode], \
        conv_scale_values = normalize_hwrf_loaded_data(hwrf_field_data[mode],
                                                  input_var_levels,
                                                  scale_format=config["conv_inputs"]["scale_format"],
                                                  scale_values=conv_scale_values)
    if not exists(config["out_path"]):
        os.makedirs(config["out_path"])
    model_objects = {}
    if args.train:
        print("Begin training")
        for model_name, model_config in config["models"].items():
            print("Training", model_name)
            model_objects[model_name] = all_models[model_config["model_type"]](**model_config["config"])
            if model_config["output_type"] == "linear":
                y_train = best_track_df["train"][output_field].values
                y_val = best_track_df["val"][output_field].values
            else:
                y_train = best_track_output_discrete["train"]
                y_val = best_track_output_discrete["train"]
            if model_config["input_type"] == "conv":
                model_objects[model_name].fit(hwrf_norm_data["train"], y_train, val_x=hwrf_norm_data["val"],
                                              val_y=y_val)
            elif model_config["input_type"] == "scalar":
                model_objects[model_name].fit(best_track_input_norm["train"].values, y_train)
            elif model_config["input_type"] == "mixed":
                model_objects[model_name].fit((best_track_input_norm["train"].values,
                                               hwrf_norm_data["train"]), y_train,
                                              val_x=(best_track_input_norm["val"].values,
                                                     hwrf_norm_data["val"]),
                                              val_y=y_val)
            print("Saving", model_name)
            tf.keras.models.save_model(model_objects[model_name].model_,
                                       join(config["out_path"], "model_path" + ".h5"),
                                       save_format="h5")
    return


if __name__ == "__main__":
    main()
