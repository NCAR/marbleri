import pandas as pd
import numpy as np
from os.path import join
random_seed = 847289
num_samples = 16384
best_track_path = "/glade/scratch/dgagne/hfip/processed_data/"
best_track_file = join(best_track_path, "best_track_all.csv")
best_track = pd.read_csv(best_track_file, index_col="Index")
np.random.seed(random_seed)
indices = np.arange(best_track.shape[0])
np.random.shuffle(indices)
best_track_sub = best_track.loc[indices[:num_samples]]
best_track_sub.to_csv(join(best_track_path, "best_track_sub.csv"), index_label="Index")
