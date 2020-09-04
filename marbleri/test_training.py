from .training import partition_storm_examples, BestTrackSequence
from .process import get_hwrf_filenames
import pandas as pd
import xarray as xr
import numpy as np
from os.path import exists, join
from os import makedirs
from .models import BaseConvNet


def test_partition_storm_examples():
    best_track_data = pd.read_csv("testdata/best_track_all.csv", index_col="Index")
    train_indices, val_indices = partition_storm_examples(best_track_data, 1, validation_proportion=0.5)
    shared = np.intersect1d(train_indices[0], val_indices[0])
    assert train_indices[0].size > 0
    assert val_indices[0].size > 0
    assert len(shared) == 0
    return


def test_best_track_sequence():
    best_track_data = pd.read_csv("testdata/best_track_test.csv", index_col="Index")
    input_cols = ["THETA_E_L100_50000", "THETA_E_L100_70000", "THETA_E_L100_85000",
                 "U_RAD_L100_50000", "U_RAD_L100_70000", "U_RAD_L100_85000",
                 "V_TAN_L100_50000", "V_TAN_L100_70000", "V_TAN_L100_85000"]
    output_col = "vmax_bt_new"
    data_width = 16
    path = "testdata/hwrf_norm"
    batch_size = 2
    def create_synthetic_netcdfs(best_track_data, input_cols, data_width, path):
        hwrf_filenames = get_hwrf_filenames(best_track_data, hwrf_path=path)
        for hwrf_filename in hwrf_filenames:
            out_data = np.random.normal(size=(len(input_cols), data_width, data_width))
            out_data[3, 5, 5] = np.nan
            out_data[4, 6, 10] = np.inf
            ds = xr.DataArray(out_data, dims=("variable", "y", "x"), coords={"variable": input_cols,
                                                                                    "y": np.arange(data_width),
                                                                                    "x": np.arange(data_width)},
                              name="hwrf_norm")
            ds.to_netcdf(hwrf_filename,
                         encoding={"hwrf_norm": {"zlib": True, "complevel": 3}}
                         )
        return

    if not exists(path):
        makedirs(path)
    create_synthetic_netcdfs(best_track_data, input_cols, data_width, path)

    return