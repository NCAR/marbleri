import argparse
import yaml
from os.path import exists, join
from os import makedirs, environ
import horovod.keras as hvd
from mpi4py import MPI
import keras.backend as K
import tensorflow as tf
import pandas as pd
from marbleri.training import partition_storm_examples, BestTrackSequence
from marbleri.models import StandardConvNet, ResNet
from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
from sklearn.preprocessing import StandardScaler

models = {"StandardConvNet": StandardConvNet,
          "ResNet": ResNet}


def main():
    # read config file
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config filepath")
    args = parser.parse_args()
    hvd.init()
    rank = hvd.rank()
    comm = MPI.COMM_WORLD
    if not exists(args.config):
        raise FileNotFoundError("Config file {0} not found.".format(args.config))
    with open(args.config, "rb") as config_file:
        config = yaml.load(config_file, yaml.Loader)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=tf_config))
    np.random.seed(config["random_seed"])
    tf.random.set_random_seed(config["random_seed"])
    # load best track data and netCDF data into memory
    best_track = pd.read_csv(config["best_track_data_path"])
    if rank == 0:
        train_rank_indices, val_rank_indices = partition_storm_examples(best_track, hvd.size(),
                                                                        config["validation_proportion"])
        all_train_indices = np.concatenate(list(train_rank_indices.values()))
        best_track_scaler = StandardScaler()
        best_track_scaler.fit(best_track.loc[all_train_indices, config["best_track_inputs"]])
    else:
        train_rank_indices = None
        val_rank_indices = None
        best_track_scaler = None
    train_rank_indices = comm.bcast(train_rank_indices, root=0)
    val_rank_indices = comm.bcast(val_rank_indices, root=0)
    best_track_scaler = comm.bcast(best_track_scaler, root=0)
    best_track_train_rank = best_track.loc[train_rank_indices[rank]]
    print("Rank", rank, train_rank_indices[rank], len(train_rank_indices[rank]))
    train_gen = BestTrackSequence(best_track_train_rank, best_track_scaler, config["best_track_inputs"], config["best_track_output"],
                                  config["hwrf_variables"], config["batch_size"], config["hwrf_norm_data_path"])
    if rank == -1:
        best_track_val_rank = best_track.loc[val_rank_indices[rank]]
        val_gen = BestTrackSequence(best_track_val_rank, best_track_scaler, config["best_track_inputs"], config["best_track_output"],
                                  config["hwrf_variables"], config["batch_size"], config["hwrf_norm_data_path"])
    else:
        val_gen = None
    # initialize neural networks
    model_objs = dict()
    for model_name, model_config in config["models"].items():
        print(model_name)
        model_objs[model_name] = models[model_config["model_type"]](**model_config["config"])
    # train neural networks
        callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
        if rank == 0:
            full_model_out_path = join(config["model_out_path"], model_name)
            if not exists(full_model_out_path):
                makedirs(full_model_out_path)
            callbacks.extend([ModelCheckpoint(join(full_model_out_path, model_name + "_e_{epoch}.h5")),
                            CSVLogger(join(full_model_out_path, model_name + "_log.csv"))])
        model_objs[model_name].fit_generator(train_gen, build=True, validation_generator=val_gen,
                                             max_queue_size=10, workers=1, use_multiprocessing=False, callbacks=callbacks)
    return


if __name__ == "__main__":
    main()
