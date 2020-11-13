from .process import *
import pandas as pd
import os
from os.path import exists


def test_get_hwrf_filenames():
    besttrack_df = pd.read_csv("testdata/best_track_all.csv", index_col="Index")
    hwrf_path = "testdata/"
    hwrf_filenames = get_hwrf_filenames(besttrack_df, hwrf_path)
    assert len(hwrf_filenames) == besttrack_df.shape[0]
    assert hwrf_path + "maria15l.2017092918.f048.nc" in hwrf_filenames


def test_coarsen_hwrf_run_set():
    hwrf_files = ["testdata/maria15l.2017092918.f036.nc", "testdata/maria15l.2017092918.f048.nc"]
    variable_levels = [("TMP_P0_L100_GLL0", 70000.0), ("TMP_P0_L103_GLL0", None)]
    out_path = "testdata/test_coarse/"
    window_size = 4
    subset_indices = (44, 556)
    if not exists(out_path):
        os.makedirs(out_path)
    coarsen_hwrf_run_set(hwrf_files, variable_levels, window_size, subset_indices, out_path)
