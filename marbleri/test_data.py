from .data import output_preds_adeck
from .nwp import BestTrackNetCDF
import pandas as pd
import numpy as np
from os.path import exists, join
from glob import glob

def test_output_preds_adeck():
    model_name = "cnn_synth_discrete"
    adeck_chars = np.array([
                   2,
                   2,
                   10,
                   2,
                   4,
                   3,
                   4,
                   5,
                   3,
                   4,
                   2,
                   3,
                   3,
                   4,
                   4,
                   4,
                   4,
                   4,
                   4,
                   3,
                   3,
                   3,
                   3,
                   3,
                   3,
                   3,
                   3,
                   10,
                   1,
                   2,
                   3,
                   4,
                   4,
                   4,
                   4], dtype=int)
    best_track_nc = BestTrackNetCDF("testdata/", file_start="diag2020", start_date="20190101", end_date="20200101")
    best_track_nc.calc_time_differences(["VMAX", "vmax_bt"], 24)
    best_track_sim = best_track_nc.to_dataframe(["VMAX_dt_24", "vmax_bt_dt_24"]).iloc[:200]
    print(best_track_sim.shape)
    pred_sim = pd.DataFrame({model_name: np.zeros(best_track_sim.shape[0]),
                            model_name + "_30": np.zeros(best_track_sim.shape[0]),
                            model_name + "_35": np.zeros(best_track_sim.shape[0]),
                            model_name + "_40": np.ones(best_track_sim.shape[0]),
                            })
    print(pred_sim.shape)
    out_path = f"testdata/{model_name}"
    output_preds_adeck(pred_sim, best_track_sim, model_name, "MCSD", out_path)
    adeck_file_list = glob(join(out_path, "*.dat"))
    assert len(adeck_file_list) > 0
    if len(adeck_file_list) > 0:
        with open(adeck_file_list[0], "r") as adeck_file:
            adeck_lines = adeck_file.readlines()
            for a, adeck_line in enumerate(adeck_lines):
                adeck_list = adeck_line.strip().split(", ")
                adeck_lengths = np.array([len(x) for x in adeck_list[:-4]], dtype=int)
                assert np.all((adeck_lengths - adeck_chars) == 0)
                # Ensure latitude conversion is correct
                lat_str = adeck_list[6]
                lat_float = float(lat_str[:-1]) / 10
                lat_float = lat_float * -1 if lat_str[-1] == "S" else lat_float
                bt_lat = best_track_sim.iloc[a]["LAT"]
                assert abs(lat_float - bt_lat) < 0.1
                # Make sure longitude conversion is correct
                lon_str = adeck_list[7]
                lon_float = float(lon_str[:-1]) / 10
                lon_float = lon_float * -1 + 360 if lon_str[-1] == "W" else lon_float
                bt_lon = best_track_sim.iloc[a]["LON"]
                assert abs(lon_float - bt_lon) < 0.1

                # Make sure predicted wind speed is positive
                assert float(adeck_list[8]) > 0



