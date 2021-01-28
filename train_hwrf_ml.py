import argparse
import yaml
from os.path import exists, join
import tensorflow as tf
from marbleri.process import get_hwrf_filenames_diff, get_var_levels, load_hwrf_data_distributed, \
    normalize_hwrf_loaded_data, discretize_output, scaler_classes
from marbleri.models import all_models
from marbleri.evaluate import linear_metrics, discrete_metrics, expected_value
from marbleri.nwp import BestTrackNetCDF
from dask.distributed import Client, LocalCluster
from marbleri.data import output_preds_adeck
import numpy as np
import pandas as pd
import os
import pickle


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
    out_path = config["out_path"]
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
    best_track_meta = {}
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
        all_meta_columns = best_track_nc[mode].run_columns + best_track_nc[mode].meta_columns
        best_track_meta[mode] = best_track_df[mode][all_meta_columns]
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
        if np.any(np.isnan(hwrf_field_data[mode])):
            print("nans:", np.where(np.isnan(hwrf_field_data[mode])))
        print("Normalize " + mode)
        hwrf_norm_data[mode], \
        conv_scale_values = normalize_hwrf_loaded_data(hwrf_field_data[mode],
                                                  input_var_levels,
                                                  scale_format=config["conv_inputs"]["scale_format"],
                                                  scale_values=conv_scale_values)
        print(conv_scale_values)
    if not exists(config["out_path"]):
        os.makedirs(config["out_path"])
    if conv_scale_values is not None:
        conv_scale_values.to_csv(join(out_path, "hwrf_conv_scale_values.csv"))
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
                y_val = best_track_output_discrete["val"]
            print(y_train)
            print(model_config["input_type"])
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
            if model_config["model_type"] == "RandomForestRegressor":
                with open(join(out_path, model_name + ".pkl"), "wb") as out_pickle:
                    pickle.dump(model_objects[model_name], out_pickle)
            else:
                tf.keras.models.save_model(model_objects[model_name].model_,
                                           join(out_path, model_name + ".h5"),
                                           save_format="h5")
            for mode in data_modes:
                if model_config["input_type"] == "scalar":
                    y_pred = model_objects[model_name].predict(best_track_input_norm[mode].values)
                elif model_config["input_type"] == "mixed":
                    y_pred = model_objects[model_name].predict((best_track_input_norm[mode].values,
                                                                     hwrf_norm_data[mode]))
                else:
                    y_pred = model_objects[model_name].predict(hwrf_norm_data[mode])
                if model_config["output_type"] == "linear":
                    y_true = best_track_df[mode][output_field].values
                    y_pred = y_pred.ravel()
                    print("y true shape", y_true.shape)
                    print("y pred shape", y_pred.shape)
                    print("y pred shape", y_pred.min(), y_pred.max())
                    model_scores = linear_metrics(y_true, y_pred, best_track_meta[mode])
                    print(f"{model_name} {mode} Linear Scores")
                    print(model_scores)
                    model_scores.to_csv(join(out_path, f"{model_name}_{mode}_linear_scores.csv"),
                                        index_label="subset")
                    pred_true_df = pd.DataFrame({output_field:y_true, model_name: y_pred},
                                                index=best_track_meta[mode].index)
                    pred_out = pd.merge(best_track_meta[mode], pred_true_df, left_index=True, right_index=True)
                    pred_out.to_csv(join(out_path, f"{model_name}_{mode}_linear_predictions.csv"))
                else:
                    y_true = best_track_df[mode][output_field].values
                    y_true_discrete = best_track_output_discrete[mode]
                    y_pred_linear = expected_value(y_pred, output_bins)
                    print("y true shape", y_true.shape)
                    print("y pred shape", y_pred.shape)
                    print("y pred linear shape", y_pred_linear.shape)
                    linear_model_scores = linear_metrics(y_true, y_pred_linear, best_track_meta[mode])
                    print(f"{model_name} {mode} Linear Scores")
                    print(linear_model_scores)
                    linear_model_scores.to_csv(join(out_path, f"{model_name}_{mode}_linear_scores.csv"),
                                        index_label="subset")
                    discrete_model_scores = discrete_metrics(y_true_discrete, y_pred, output_bins,
                                                             best_track_meta[mode])
                    print(f"{model_name} {mode} Discrete Scores")
                    print(discrete_model_scores)
                    discrete_model_scores.to_csv(join(out_path, f"{model_name}_{mode}_discrete_scores.csv"),
                                                 index_label="subset")
                    pred_true_dict = {f"{model_name}_{o_bin:02.0f}": y_pred[:, y_pred.shape[1] - 1 - o]
                                      for o, o_bin in enumerate(output_bins[::-1])}
                    pred_true_dict.update({output_field: y_true, model_name: y_pred_linear})
                    pred_true_df = pd.DataFrame(pred_true_dict,
                                                index=best_track_meta[mode].index)
                    pred_out = pd.merge(best_track_meta[mode], pred_true_df, left_index=True, right_index=True)
                    pred_out.to_csv(join(out_path, f"{model_name}_{mode}_discrete_predictions.csv"))
                if "adeck_name" not in model_config.keys():
                    adeck_name = "M" + "".join([x[0] for x in model_name.split("_")]).upper()
                else:
                    adeck_name = model_config["adeck_name"]
                if mode != "train":
                    adeck_out_dir = join(out_path, "adeck_" + model_name, mode)
                    if not exists(adeck_out_dir):
                        os.makedirs(adeck_out_dir)
                    output_preds_adeck(pred_out, best_track_df[mode], model_name, adeck_name, adeck_out_dir)
    return


if __name__ == "__main__":
    main()
