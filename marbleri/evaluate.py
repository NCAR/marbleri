import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from functools import partial


def linear_metrics(y_true, y_pred, meta):
    metric_set = {"RMSE": root_mean_squared_error,
                  "MAE": mean_absolute_error,
                  "ME": mean_error,
                  "R": corr_coef,
                  "AE_67": absolute_error_percentile,
                  "AE_90": partial(absolute_error_percentile, percentile=90)
                  }
    subsets = ["all"]
    basins = np.unique(meta["BASIN"])
    forecast_hours = np.unique(meta["TIME"])
    for basin in basins:
        for forecast_hour in forecast_hours:
            subsets.append(f"{basin}_f{forecast_hour:03d}")
    all_scores = pd.DataFrame(0.0, index=subsets, columns=list(metric_set.keys()) + ["Count"])
    for metric, metric_fun in metric_set.items():
        all_scores.loc["all", metric] = metric_fun(y_true, y_pred)
        all_scores.loc["all", "Count"] = y_true.size
    for basin in basins:
        for forecast_hour in forecast_hours:
            subset = f"{basin}_f{forecast_hour:03d}"
            sub_i = (meta["BASIN"] == basin) & (meta["TIME"] == forecast_hour)
            all_scores.loc[subset, "Count"] = np.count_nonzero(sub_i)
            if all_scores.loc[subset, "Count"] > 0:
                for metric, metric_fun in metric_set.items():
                    all_scores.loc[subset, metric] = metric_fun(y_true[sub_i], y_pred[sub_i])
            else:
                all_scores.loc[subset, list(metric_set.keys())] = np.nan
    return all_scores


def discrete_metrics(y_true_discrete, y_pred_discrete, y_bins, meta):
    metric_set = {"RPS": ranked_probability_score,
                  "BS_0": partial(brier_score_threshold, threshold=0),
                  "BS_20": partial(brier_score_threshold, threshold=20),
                  "BS_30": partial(brier_score_threshold, threshold=30),
                  "BS_35": partial(brier_score_threshold, threshold=35),
                  "BSS_0": partial(brier_score_threshold, threshold=0),
                  "BSS_20": partial(brier_score_threshold, threshold=20),
                  "BSS_30": partial(brier_score_threshold, threshold=30),
                  "BSS_35": partial(brier_score_threshold, threshold=35),
                  "AUC_0": partial(auc_threshold, threshold=0),
                  "AUC_20": partial(auc_threshold, threshold=20),
                  "AUC_30": partial(auc_threshold, threshold=30),
                  "AUC_35": partial(auc_threshold, threshold=35)}
    subsets = ["all"]
    basins = np.unique(meta["BASIN"])
    forecast_hours = np.unique(meta["TIME"])
    for basin in basins:
        for forecast_hour in forecast_hours:
            subsets.append(f"{basin}_f{forecast_hour:03d}")
    all_scores = pd.DataFrame(0.0, index=subsets, columns=list(metric_set.keys()) + ["Count"])
    for metric, metric_fun in metric_set.items():
        print(metric)
        print(metric_fun(y_true_discrete, y_pred_discrete, y_bins))
        all_scores.loc["all", metric] = metric_fun(y_true_discrete, y_pred_discrete, y_bins)
        all_scores.loc["all", "Count"] = y_true_discrete.shape[0]
    for basin in basins:
        for forecast_hour in forecast_hours:
            subset = f"{basin}_f{forecast_hour:03d}"
            sub_i = (meta["BASIN"] == basin) & (meta["TIME"] == forecast_hour)
            all_scores.loc[subset, "Count"] = np.count_nonzero(sub_i)
            if all_scores.loc[subset, "Count"] > 0:
                for metric, metric_fun in metric_set.items():
                    all_scores.loc[subset, metric] = metric_fun(y_true_discrete[sub_i],
                                                                y_pred_discrete[sub_i],
                                                                y_bins)
            else:
                all_scores.loc[subset, list(metric_set.keys())] = np.nan
    return all_scores


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_error(y_true, y_pred):
    return np.mean(y_pred - y_true)


def corr_coef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def absolute_error_percentile(y_true, y_pred, percentile=67):
    return np.percentile(np.abs(y_pred - y_true), percentile)


def exceedance_probability(y_discrete, y_bins, threshold):
    b_index = np.searchsorted(y_bins, threshold)
    return y_discrete[:, b_index:].sum(axis=1)


def ranked_probability_score(y_true_discrete, y_pred_discrete, y_bins):
    y_pred_cumulative = np.cumsum(y_pred_discrete)
    y_true_cumulative = np.cumsum(y_true_discrete)
    return np.mean((y_pred_cumulative - y_true_cumulative) ** 2) / float(y_pred_discrete.shape[1] - 1)


def brier_score_threshold(y_true_discrete, y_pred_discrete, y_bins, threshold=0):
    y_pred_prob = exceedance_probability(y_pred_discrete, y_bins, threshold)
    y_true = exceedance_probability(y_true_discrete, y_bins, threshold)
    return np.mean((y_pred_prob - y_true) ** 2)


def brier_skill_score_threshold(y_true_discrete, y_pred_discrete, y_bins, threshold=0):
    y_pred_prob = exceedance_probability(y_pred_discrete, y_bins, threshold)
    y_true = exceedance_probability(y_true_discrete, y_bins, threshold)
    bs = np.mean((y_pred_prob - y_true) ** 2)
    bs_c = np.mean((y_true.mean() - y_true) ** 2)
    return 1 - bs / bs_c


def auc_threshold(y_true_discrete, y_pred_discrete, y_bins, threshold=0):
    y_pred_prob = exceedance_probability(y_pred_discrete, y_bins, threshold)
    y_true = exceedance_probability(y_true_discrete, y_bins, threshold)
    if len(np.unique(y_true)) > 1:
        score = roc_auc_score(y_true, y_pred_prob)
    else:
        score = np.nan
    return score


def expected_value(y_pred_discrete, y_bins):
    y_bin_centers = np.zeros(y_bins.size)
    y_bin_centers[:-1] = 0.5 * (y_bins[:-1] + y_bins[1:])
    y_bin_centers[-1] = y_bins[-1]
    return np.sum(y_pred_discrete * y_bins, axis=1)


