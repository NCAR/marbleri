import argparse
import os
import pickle
from os.path import exists, join

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dask.distributed import Client, LocalCluster

from marbleri.data import output_preds_adeck
from marbleri.evaluate import linear_metrics, discrete_metrics, expected_value
from marbleri.models import all_models
from marbleri.nwp import BestTrackNetCDF
from marbleri.process import get_hwrf_filenames_diff, get_var_levels, load_hwrf_data_distributed, \
    normalize_hwrf, discretize_output, scaler_classes


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
    if len(gpus) > 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    data_modes = config["data_modes"]
    conv_subset = config["conv_inputs"]["subset"]
    out_path = config["out_path"]
    any_conv_models = any([(model_config["input_type"] == "conv") or
                           (model_config["input_type"] == "mixed") for model_config in config["models"].values()])
    best_track_nc = {}
    best_track_df = {}
    dt = config["time_difference_hours"]
    best_track_inputs_static = config["best_track_inputs_static"]
    best_track_inputs_dt = [f"{bt}_dt_{dt:02d}" for bt in config["best_track_inputs_dt"]]
    best_track_inputs_ml = best_track_inputs_static + best_track_inputs_dt
    best_track_input_norm = {}
    input_var_levels = get_var_levels(config["conv_inputs"]["variables"], config["conv_inputs"]["levels"])
    output_field = config["best_track_output"] + f"_dt_{dt:02d}"
    hwrf_field_data = {}
    output_bins = np.arange(config["output_bins"][0],
                            config["output_bins"][1] + config["output_bins"][2],
                            config["output_bins"][2])

    best_track_output_discrete = {}
    best_track_meta = {}
    conv_scale_values = None
    for mode in data_modes:
        print("Loading " + mode)
        best_track_nc[mode] = BestTrackNetCDF(**config["best_track_data_paths"][mode])
        best_track_nc[mode].calc_time_differences(config["best_track_inputs_dt"], config["time_difference_hours"])
        best_track_nc[mode].calc_time_differences([config["best_track_output"]], config["time_difference_hours"])
        best_track_df[mode] = best_track_nc[mode].to_dataframe(best_track_inputs_static +
                                                               best_track_inputs_dt + [output_field])
        best_track_df[mode].to_csv(join(out_path, f"best_track_{mode}.csv"))
        all_meta_columns = best_track_nc[mode].run_columns + best_track_nc[mode].meta_columns
        best_track_meta[mode] = best_track_df[mode][all_meta_columns]
        best_track_output_discrete[mode] = discretize_output(best_track_df[mode][output_field].values, output_bins)
        print(best_track_df[mode])
    if any_conv_models:
        conv_scale_values = []
        cluster = LocalCluster(n_workers=config["n_workers"], threads_per_worker=1)
        client = Client(cluster)
        for mode in data_modes:
            hwrf_filenames_start, hwrf_filenames_end = get_hwrf_filenames_diff(best_track_df[mode],
                                                                               hwrf_data_paths[mode],
                                                                               diff=config["time_difference_hours"])
            print(hwrf_filenames_start[0], hwrf_filenames_end[0])
            hwrf_files_se = np.vstack([hwrf_filenames_start, hwrf_filenames_end]).T
            hwrf_field_data[mode] = load_hwrf_data_distributed(hwrf_files_se, input_var_levels, client,
                                                               subset=conv_subset)
            if np.any(np.isnan(hwrf_field_data[mode])):
                print("nans:", np.where(np.isnan(hwrf_field_data[mode])))
            # print("Normalize " + mode)
            # hwrf_norm_data[mode], \
            # conv_scale_values = normalize_hwrf_loaded_data(hwrf_field_data[mode],
            #                                        input_var_levels,
            #                                        scale_format=config["conv_inputs"]["scale_format"],
            #                                        scale_values=conv_scale_values)
            # print(conv_scale_values)
    if not exists(out_path):
        os.makedirs(out_path)
    # if conv_scale_values is not None:
    #    conv_scale_values.to_csv(join(out_path, "hwrf_conv_scale_values.csv"))
    model_objects = {}
    for model_name in config["models"].keys():
        model_objects[model_name] = []
    if args.train:
        print("Begin training")
        forecast_hours = np.unique(best_track_df["train"]["TIME"])
        best_track_scalers = [scaler_classes[config["best_track_scaler"]]() for fh in forecast_hours]
        print("Forecast Hours: ", forecast_hours)
        for f, forecast_hour in enumerate(forecast_hours):
            fh_indices = best_track_df["train"]["HOUR"] == forecast_hour
            best_track_in_hour_norm = pd.DataFrame(best_track_scalers[f].
                                                   fit_transform(best_track_df["train"].loc[fh_indices,
                                                                                            best_track_inputs_ml].values),
                                                   columns=best_track_inputs_ml)
            if any_conv_models:
                conv_scale_values.append(None)
                hwrf_norm_hour, conv_scale_values[f] = normalize_hwrf(hwrf_field_data["train"][fh_indices],
                                                                      input_var_levels,
                                                                      scale_format=config["conv_inputs"][
                                                                          "scale_format"],
                                                                      scale_values=conv_scale_values[f])
            else:
                hwrf_norm_hour = None
            for model_name, model_config in config["models"].items():
                if model_config["output_type"] == "linear":
                    y_train = best_track_df["train"].loc[fh_indices, output_field].values
                    y_val = best_track_df["val"].loc[fh_indices, output_field].values
                else:
                    y_train = best_track_output_discrete["train"].loc[fh_indices].values
                    y_val = best_track_output_discrete["val"].loc[fh_indices].values
                print("Training", model_name, forecast_hour)
                model_objects[model_name].append(all_models[model_config["model_type"]](**model_config["config"]))
                if model_config["input_type"] == "conv":
                    model_objects[model_name][-1].fit(hwrf_norm_hour, y_train)
                elif model_config["input_type"] == "scalar":
                    model_objects[model_name][-1].fit(best_track_in_hour_norm.values, y_train)
                elif model_config["input_type"] == "mixed":
                    model_objects[model_name][-1].fit((best_track_in_hour_norm.values,
                                                       hwrf_norm_hour), y_train)
                print("Saving", model_name)
                if model_config["model_type"] == "RandomForestRegressor":
                    with open(join(out_path, f"{model_name}_f{forecast_hour:03d}.pkl"), "wb") as out_pickle:
                        pickle.dump(model_objects[model_name][f], out_pickle)
                else:
                    if hasattr(model_objects[model_name][f], "model_"):
                        tf.keras.models.save_model(model_objects[model_name][f].model_,
                                                   join(out_path, f"{model_name}_f{forecast_hour:03d}.h5"),
                                                   save_format="h5")
                    else:
                        print(f"{model_name} at f{forecast_hour:03d} was not trained and cannot be saved.")
        print("Validating trained models")
        for mode in data_modes:
            all_preds = {}
            for model_name, model_config in config["models"].items():
                if model_config["output_type"] == "linear":
                    all_preds[model_name] = pd.DataFrame(0.0,
                                                         columns=[output_field, model_name],
                                                         index=best_track_meta[mode].index)
                    all_preds[model_name].loc[:, output_field] = best_track_df[mode][output_field].values

                else:
                    out_bin_model = [f"{model_name}_{o_bin:02.0f}" for o_bin in output_bins[::-1]]
                    cols = [output_field, model_name] + out_bin_model
                    all_preds[model_name] = pd.DataFrame(0.0,
                                                         columns=cols,
                                                         index=best_track_meta[mode].index)
                    all_preds[model_name].loc[:, output_field] = best_track_df[mode][output_field].values

            for f, forecast_hour in enumerate(forecast_hours):
                print(f"Validating f{forecast_hour:03d}")
                fh_indices = best_track_df[mode]["HOUR"] == forecast_hour
                best_track_in_hour_norm = pd.DataFrame(best_track_scalers[f].
                                                       transform(best_track_df[mode].loc[fh_indices,
                                                                                         best_track_inputs_ml].values),
                                                       columns=best_track_inputs_ml)
                if any_conv_models:
                    hwrf_norm_hour, _ = normalize_hwrf(hwrf_field_data[mode][fh_indices], input_var_levels,
                                                       scale_format=config["conv_inputs"]["scale_format"],
                                                       scale_values=conv_scale_values[f])
                else:
                    hwrf_norm_hour = None
                for model_name, model_config in config["models"].items():
                    out_bin_model = [f"{model_name}_{o_bin:02.0f}" for o_bin in output_bins[::-1]]
                    if model_config["input_type"] == "conv":
                        y_pred = model_objects[model_name][f].predict(hwrf_norm_hour)
                    elif model_config["input_type"] == "mixed":
                        y_pred = model_objects[model_name][f].predict((best_track_input_norm.values,
                                                                       hwrf_norm_hour))
                    else:
                        y_pred = model_objects[model_name][f].predict(best_track_in_hour_norm.values)
                    if model_config["output_type"] == "linear":
                        all_preds[model_name].loc[fh_indices, model_name] = y_pred.ravel()
                    else:
                        y_pred_linear = expected_value(y_pred, output_bins)
                        all_preds[model_name].loc[fh_indices, model_name] = y_pred_linear
                        all_preds[model_name].loc[fh_indices, out_bin_model] = y_pred
            for model_name, model_config in config["models"].items():
                out_bin_model = [f"{model_name}_{o_bin:02.0f}" for o_bin in output_bins[::-1]]
                linear_model_scores = linear_metrics(all_preds[model_name][output_field].values,
                                                     all_preds[model_name][model_name].values,
                                                     best_track_meta[mode])
                print(f"{model_name} {mode} Linear Scores")
                print(linear_model_scores)
                linear_model_scores.to_csv(join(out_path, f"{model_name}_{mode}_linear_scores.csv"),
                                           index_label="subset")
                if model_config["output_type"] == "discrete":
                    discrete_model_scores = discrete_metrics(best_track_output_discrete[mode],
                                                             all_preds[model_name][out_bin_model],
                                                             output_bins,
                                                             best_track_meta[mode])
                    print(f"{model_name} {mode} Discrete Scores")
                    print(discrete_model_scores)
                    discrete_model_scores.to_csv(join(out_path, f"{model_name}_{mode}_discrete_scores.csv"),
                                                 index_label="subset")
                pred_out = pd.merge(best_track_meta[mode], all_preds[model_name],
                                    left_index=True, right_index=True)
                pred_out.to_csv(join(out_path, f"{model_name}_{mode}_discrete_predictions.csv"))
                if "adeck_name" not in model_config.keys():
                    adeck_name = "M" + "".join([x[0] for x in model_name.split("_")]).upper()
                else:
                    adeck_name = model_config["adeck_name"]
                if mode != "train":
                    adeck_out_dir = join(out_path, "adeck_" + model_name, mode)
                    if not exists(adeck_out_dir):
                        os.makedirs(adeck_out_dir)
                    output_preds_adeck(pred_out, best_track_df[mode], model_name, adeck_name, adeck_out_dir,
                                       time_difference_hours=config["time_difference_hours"])
    return


if __name__ == "__main__":
    main()
