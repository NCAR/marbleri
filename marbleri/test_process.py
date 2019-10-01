from .process import *
import pandas as pd
import os
from dask.distributed import LocalCluster, Client
from os.path import exists


def test_get_hwrf_filenames():
    besttrack_df = pd.read_csv("testdata/best_track_all.csv", index_col="Index")
    hwrf_path = "testdata/"
    hwrf_filenames = get_hwrf_filenames(besttrack_df, hwrf_path)
    assert len(hwrf_filenames) == besttrack_df.shape[0]
    assert hwrf_path + "maria15l.2017092918.f048.nc" in hwrf_filenames


def test_hwrf_step_local_sums():
    hwrf_filename = "testdata/maria15l.2017092918.f048.nc"
    variable_levels = [("TMP_P0_L100_GLL0", 70000.0), ("TMP_P0_L100_GLL0", None)]
    subset_indices = [108, 492]
    local_sum = hwrf_step_local_sums(hwrf_filename, variable_levels, subset_indices)
    assert len(local_sum.shape) == 4
    assert local_sum.shape[0] == len(variable_levels)
    assert local_sum.shape[-1] == 2
    assert local_sum[:, :, :, 1].max() == 1
    assert local_sum[:, :, :, 1].min() >= 0



def test_hwrf_step_local_variances():
    hwrf_filename = "testdata/maria15l.2017092918.f048.nc"
    variable_levels = [("TMP_P0_L100_GLL0", 70000.0), ("TMP_P0_L100_GLL0", None)]
    subset_indices = [108, 492]

    local_sum = hwrf_step_local_sums(hwrf_filename, variable_levels, subset_indices)
    local_means = local_sum[:, :, :, 0] / local_sum[:, :, :, 1]
    local_var = hwrf_step_local_variances(hwrf_filename, variable_levels, local_means, subset_indices)
    assert local_var.shape[0] == len(variable_levels)
    assert local_var.shape[-1] == 2
    assert len(local_var.shape) == 4
    assert local_var[:, :, :, 0].sum() == 0


def test_process_hwrf_run_set():
    hwrf_files = ["testdata/maria15l.2017092918.f048.nc", "testdata/maria15l.2017092918.f036.nc"]
    variable_levels = [("TMP_P0_L100_GLL0", 70000.0), ("TMP_P0_L103_GLL0", None)]
    out_path = "testdata/testout/"
    if not exists(out_path):
        os.makedirs(out_path)
    n_workers = 2
    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster)
    subset_indices = [108, 492]
    norm_values = calculate_hwrf_local_norms(hwrf_files, variable_levels, subset_indices, out_path, client, n_workers)
    assert exists(join(out_path, "hwrf_local_norm_stats.nc"))
    out = process_hwrf_run_set(hwrf_files, variable_levels, subset_indices, out_path, norm_values, global_norm=False)
    out_files = [out_path + x.split("/")[-1] for x in hwrf_files]
    assert out == 0
    assert exists(out_files[0])
    assert exists(out_files[1])
    for out_file in out_files:
        if exists(out_file):
            os.remove(out_file)
    os.remove(out_path + "hwrf_local_norm_stats.nc")
    os.rmdir(out_path)
    client.close()


def test_coarsen_hwrf_run_set():
    hwrf_files = ["testdata/maria15l.2017092918.f036.nc", "testdata/maria15l.2017092918.f048.nc"]
    variable_levels = [("TMP_P0_L100_GLL0", 70000.0), ("TMP_P0_L103_GLL0", None)]
    out_path = "testdata/test_coarse/"
    window_size = 4
    subset_indices = (44, 556)
    if not exists(out_path):
        os.makedirs(out_path)
    coarsen_hwrf_run_set(hwrf_files, variable_levels, window_size, subset_indices, out_path)
