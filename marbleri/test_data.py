from .data import output_preds_adeck
import pandas as pd
import numpy as np
from os.path import exists, join
from glob import glob

def test_output_preds_adeck():
    model_name = "cnn_synth_discrete"
    adeck_chars = np.array([2,
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
    best_track_sim = pd.read_csv("testdata/best_track_all.csv", nrows=200)
    pred_sim = pd.DataFrame({model_name: np.random.normal(scale=20, size=best_track_sim.shape[0]),
                            model_name + "_30": np.zeros(best_track_sim.shape[0]),
                            model_name + "_35": np.zeros(best_track_sim.shape[0]),
                            model_name + "_40": np.ones(best_track_sim.shape[0]),
                            })
    best_track_sim["VMAX_dt_24"] = best_track_sim["VMAX"]
    out_path = f"testdata/{model_name}"
    output_preds_adeck(pred_sim, best_track_sim, model_name, "MCSD", out_path)
    adeck_file_list = glob(join(out_path, "*.dat"))
    assert len(adeck_file_list) > 0
    if len(adeck_file_list) > 0:
        with open(adeck_file_list[0], "r") as adeck_file:
            adeck_lines = adeck_file.readlines()
            for adeck_line in adeck_lines:
                adeck_list = adeck_line.strip().split(", ")
                adeck_lengths = np.array([len(x) for x in adeck_list[:-4]], dtype=int)
                assert np.all((adeck_lengths - adeck_chars) == 0)



