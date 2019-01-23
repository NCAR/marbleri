import xarray as xr
from os.path import exists


class HWRFStep(object):
    def __init__(self, filename):
        if not exists(filename):
            raise FileNotFoundError(filename + " not found.")
        self.filename = filename
        self.ds = xr.open_dataset(filename)
        self.levels = self.ds["lv_ISBL0"].values

    def get_variable(self, variable_name, level=None, subset=None):
        if variable_name not in list(self.ds.variables):
            raise KeyError(variable_name + " not available in " + self.filename)
        if subset is None:
            subset_y = slice(0, self.ds.dims["lat_0"])
            subset_x = slice(0, self.ds.dims["lon_0"])
        else:
            subset_y = slice(subset[0], subset[1])
            subset_x = slice(subset[0], subset[1])
        if level is None:
            if len(self.ds.variable.shape) == 3:
                level_index = 0
            else:
                level_index = -1
        else:
            if level in self.levels:
                level_index = np.where(self.levels == level)[0][0]
            else:
                raise KeyError("level {0} not found.".format(level))
        if level_index >= 0:
            var_data = self.ds[variable_name][level_index, subset_y, subset_x]
        else:
            var_data = self.ds[variable_name][subset_y, subset_x]
        return var_data

    def close(self):
        self.ds.close()






