import numpy as np
from keras.utils import Sequence
from .nwp import HWRFStep


class HWRFSequence(Sequence):
    """


    """
    def __init__(self, hwrf_files, best_track, hwrf_variables, hwrf_variable_levels,
                 best_track_input_variables, best_track_label_variables, batch_size,
                 x_start=0, x_end=601, shuffle=True):
        self.hwrf_files = hwrf_files
        self.best_track = best_track
        self.hwrf_variables = hwrf_variables
        self.hwrf_variable_levels = hwrf_variable_levels
        self.num_hwrf_variables = len(self.hwrf_variables)
        self.best_track_input_variables = best_track_input_variables
        self.num_best_track_input_variables = len(best_track_input_variables)
        self.best_track_label_variables = best_track_label_variables
        self.num_best_track_label_variables = len(best_track_label_variables)
        self.batch_size = batch_size
        self.x_subset = (x_start, x_end)
        self.x_size = x_end - x_start
        self.shuffle = True
        self.indexes = np.arange(len(self.hwrf_files))
        self.on_epoch_end()
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.hwrf_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.hwrf_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_list_temp):
        all_hwrf_patches = np.zeros((index_list_temp.size, self.x_size, self.x_size, self.num_hwrf_variables), dtype=np.float32)
        all_best_track_input_data = np.zeros((index_list_temp.size, self.num_best_track_input_variables), dtype=np.float32)
        all_best_track_label_data = np.zeros((index_list_temp.size, self.num_best_track_label_variables), dtype=np.float32)
        for i, idx in enumerate(index_list_temp):
            hwrf_file = self.hwrf_files[idx]
            hwrf_data = HWRFStep(hwrf_file)
            for v, variable in enumerate(self.hwrf_variables):
                all_hwrf_patches[i, :, :, v] = hwrf_data.get_variable(variable, level=self.hwrf_variable_levels[v],
                                                            subset=self.x_subset)
                all_hwrf_patches[i, :, :, v][np.isnan(all_hwrf_patches[i, :, :, v])] = np.nanmin(all_hwrf_patches[i, :, :, v])
            hwrf_data.close()
            all_best_track_input_data[i] = self.best_track.get_storm_variables(self.best_track_input_variables,
                                            hwrf_data.run_date, hwrf_data.storm_name, hwrf_data.storm_number,
                                            hwrf_data.basin, hwrf_data.forecast_hour)
            all_best_track_label_data[i] = self.best_track.get_storm_variables(self.best_track_label_variables,
                                                                    hwrf_data.run_date, hwrf_data.storm_name,
                                                                    hwrf_data.storm_number,
                                                                    hwrf_data.basin, hwrf_data.forecast_hour)
        return [all_hwrf_patches, all_best_track_input_data], all_best_track_label_data
