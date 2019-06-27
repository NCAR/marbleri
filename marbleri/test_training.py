from .training import partition_storm_examples
import pandas as pd
import numpy as np


def test_partition_storm_examples():
    best_track_data = pd.read_csv("testdata/best_track_all.csv", index_col="Index")
    train_indices, val_indices = partition_storm_examples(best_track_data, 1, validation_proportion=0.5)
    shared = np.intersect1d(train_indices[0], val_indices[0])
    assert train_indices[0].size > 0
    assert val_indices[0].size > 0
    assert len(shared) == 0
    return