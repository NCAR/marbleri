import numpy as np
from keras.utils import Sequence
from .nwp import HWRFStep
from .process import get_hwrf_filenames
import xarray as xr


def partition_storm_examples(best_track_data, num_ranks, validation_proportion=0.2):
    """
    Separates data into training and validation sets by storm name.

    Args:
        best_track_data: `pandas.DataFrame` containing best track storm data
        num_ranks: Number of MPI ranks (processes)
        validation_proportion: Proportion from 0 to 1 of the best_track examples to be used for validation

    Returns:
        train_rank_indices: dictionary of best_track_data indices for each MPI rank
    """
    full_names = best_track_data["STNAM"].astype(str) + best_track_data["STNUM"].astype(str) \
        + best_track_data["BASIN"].astype(str) + best_track_data["DATE"].astype(str).str[:4]
    unique_names = np.unique(full_names.values)
    num_unique = unique_names.size
    num_validation = int(num_unique * validation_proportion)
    num_train = num_unique - num_validation
    name_indices = np.random.permutation(np.arange(unique_names.shape[0]))
    train_names = unique_names[name_indices[:num_train]]
    val_names = unique_names[name_indices[:num_train]]
    train_indices = np.where(np.in1d(full_names.values, train_names))[0]
    val_indices = np.where(np.in1d(full_names.values, val_names))[0]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    train_rank_indices = {}
    val_rank_indices = {0: val_indices}
    train_rank_size = train_indices.size // num_ranks
    val_rank_size = val_indices.size // num_ranks
    for i in range(num_ranks):
        train_rank_indices[i] = train_indices[i * train_rank_size: (i + 1) * train_rank_size]
        if i > 0:
            val_rank_indices[i] = None
    return train_rank_indices, val_rank_indices


class BestTrackSequence(Sequence):
    def __init__(self, best_track_data, best_track_scaler, best_track_inputs, best_track_output,
                 hwrf_inputs, batch_size, hwrf_path, shuffle=True, data_format="channels_first", domain_width=384):
        self.best_track_data = best_track_data.reset_index()
        self.best_track_scaler = best_track_scaler
        self.best_track_inputs = best_track_inputs
        self.best_track_output = best_track_output
        self.hwrf_inputs = hwrf_inputs
        self.hwrf_path = hwrf_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.domain_width = domain_width
        self.data_format = data_format
        if self.data_format == "channels_first":
            self.conv_batch_shape = (self.batch_size, len(self.hwrf_inputs), self.domain_width, self.domain_width)
        else:
            self.conv_batch_shape = (self.batch_size, self.domain_width, self.domain_width, len(self.hwrf_inputs))
        self.best_track_norm = self.best_track_scaler.transform(self.best_track_data[self.best_track_inputs])
        self.indices = np.arange(self.best_track_data.shape[0])
        self.hwrf_file_names = get_hwrf_filenames(self.best_track_data, self.hwrf_path, ".nc")
        if self.data_format == "channels_first":
            self.conv_inputs = np.zeros((self.hwrf_file_names.size, len(self.hwrf_inputs), self.domain_width,
                                     self.domain_width),
                                     dtype=np.float32)
        else:
            self.conv_inputs = np.zeros((self.hwrf_file_names.size, self.domain_width,
                                     self.domain_width, len(self.hwrf_inputs)),
                                     dtype=np.float32)
        for h, hwrf_file_name in enumerate(self.hwrf_file_names):
            if h % 100 == 0:
                print(h * 100 / self.hwrf_file_names.size, hwrf_file_name)
            hwrf_ds = xr.open_dataset(hwrf_file_name, decode_cf=False, decode_coords=False, decode_times=False,
                                      autoclose=False)
            if self.data_format == "channels_last":
                self.conv_inputs[h] = hwrf_ds["hwrf_norm"].transpose("lat", "lon", "variable").values
            else:
                self.conv_inputs[h] = hwrf_ds["hwrf_norm"].values
            hwrf_ds.close()
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(self.indices.size / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        scalar_inputs = self.best_track_norm[batch_indices]
        output = self.best_track_data.loc[batch_indices, self.best_track_output].values
        batch_conv_inputs = self.conv_inputs[batch_indices]
        return [scalar_inputs, batch_conv_inputs], output

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)





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


