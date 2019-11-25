import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from tensorflow.keras.models import Model
import numpy as np
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(tf.config.experimental.list_physical_devices())
except AttributeError:
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=tf_config))
    from tensorflow.python.client import device_lib
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    print(get_available_gpus())
i = Input((8, 8, 2))
c = Conv2D(8, (3, 3), padding="same", activation="relu")(i)
f = Flatten()(c)
o = Dense(1)(f)
m = Model(i, o)
m.compile("adam", "mse")
x = np.random.normal(size=(256, 8, 8, 2))
y = np.random.normal(size=(256,))
m.fit(x, y)
