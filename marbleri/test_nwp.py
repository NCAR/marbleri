from .nwp import BestTrackNetCDF, HWRFStep
import pandas as pd
import numpy as np

def test_hwrfstep():
    filename = "testdata/maria15l.2017092918.f036.nc"
    hwrf_step_obj = HWRFStep(filename)
    assert hwrf_step_obj.storm_name == "maria"
    assert hwrf_step_obj.storm_number == "15"
    assert hwrf_step_obj.basin == "l"
    assert hwrf_step_obj.run_date_str == "2017092918"
    assert hwrf_step_obj.run_date.year == 2017
    assert hwrf_step_obj.run_date.month == 9
    assert hwrf_step_obj.run_date.day == 29
    assert hwrf_step_obj.run_date.hour == 18
    assert hwrf_step_obj.forecast_hour == 36
    return


def test_bestracknetcdf():
    file_path = "testdata"
    file_start = "diag2020"
    start_date_str = "2018-01-01"
    end_date_str = "2019-01-01"
    bt_nc = BestTrackNetCDF(file_path, file_start, start_date_str, end_date_str)
    assert len(bt_nc.best_track_files) == 1
    assert bt_nc.bt_runs.shape[0] == bt_nc.bt_ds.run.size
    run_dates = pd.DatetimeIndex(bt_nc.bt_runs["DATE"] + "00")
    assert np.all((run_dates >= start_date_str) & (run_dates <= end_date_str))
    assert "nrun" not in bt_nc.bt_ds.dims.keys()
    td_vars = ["VMAX", "LAND"]
    time_diff = 24
    bt_nc.calc_time_differences(td_vars, time_diff)
    bt_nc.bt_ds.variables.keys()
    for td_var in td_vars:
        assert td_var + "_dt_{0:02d}".format(time_diff) in bt_nc.bt_ds.variables.keys()
    bt_df = bt_nc.to_dataframe(td_vars)
    assert "TIME" in bt_df.columns
    for td_var in td_vars:
        assert td_var in bt_df.columns
    bt_nc.close()
    return
