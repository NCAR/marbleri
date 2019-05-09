from .process import *

def test_hwrf_step_local_sums():
    hwrf_filename = "/glade/p/ral/nsap/rozoff/hfip/reforecast/maria15l.2017092918.f048.nc"
    variable_levels = [("TMP_P0_L100_GLL0", 70000.0), ("TMP_P0_L100_GLL0", None)]
    local_sum = hwrf_step_local_sums(hwrf_filename, variable_levels)
    assert local_sum.shape[0] == len(variable_levels)
    assert local_sum.shape[-1] == 2

