from .data import output_preds_adeck
import pandas as pd
import numpy as np
from os.path import exists, join
from glob import glob

def test_output_preds_adeck():
    model_name = "cnn_synth_discrete"
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
