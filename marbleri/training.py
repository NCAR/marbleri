import numpy as np
from keras.utils import Sequence
from .process import get_hwrf_filenames
import xarray as xr
from os.path import exists
import logging


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
    val_names = unique_names[name_indices[num_train:]]
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
    """
    Generator for HWRF convolutional neural network training.

    """
    def __init__(self, best_track_data, best_track_scaler, best_track_inputs, best_track_output,
                 hwrf_inputs, batch_size, hwrf_path,
                 conv_only=True, shuffle=True, data_format="channels_first", domain_width=384):
        self.best_track_data = best_track_data.reset_index()
        self.best_track_scaler = best_track_scaler
        self.best_track_inputs = best_track_inputs
        self.best_track_output = best_track_output
        self.conv_only = conv_only
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
        if len(self.best_track_inputs) > 0:
            self.best_track_norm = self.best_track_scaler.transform(self.best_track_data[self.best_track_inputs]).astype(np.float32)
        else:
            self.best_track_norm = None
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
            if exists(hwrf_file_name):
                hwrf_ds = xr.open_dataset(hwrf_file_name, decode_cf=False, decode_coords=False, decode_times=False)
                if self.data_format == "channels_last":
                    self.conv_inputs[h] = hwrf_ds["hwrf_norm"].transpose("lat", "lon", "variable").values
                else:
                    self.conv_inputs[h] = hwrf_ds["hwrf_norm"].values
                hwrf_ds.close()
            else:
                logging.info(hwrf_file_name + " does not exist")
        nan_points = np.count_nonzero(~np.isfinite(self.conv_inputs))
        if nan_points > 0:
            print(f"Number of nan points: {nan_points:d}") 
            self.conv_inputs[~np.isfinite(self.conv_inputs)] = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(self.indices.size / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        output = self.best_track_data.loc[batch_indices, self.best_track_output].values.astype(np.float32)
        batch_conv_inputs = self.conv_inputs[batch_indices]
        if not self.conv_only:
            scalar_inputs = self.best_track_norm[batch_indices]
            gen_out = ([scalar_inputs, batch_conv_inputs], output)
        else:
            gen_out = (batch_conv_inputs, output)
        return gen_out

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

