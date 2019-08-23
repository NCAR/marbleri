from keras.layers import Dense, Conv2D, Activation, Input, Flatten, AveragePooling2D, MaxPool2D, LeakyReLU, Dropout, Add
from keras.layers import BatchNormalization, Concatenate, Layer, SpatialDropout2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error
from keras.utils import multi_gpu_model
from keras.regularizers import l2
from functools import partial
import numpy as np
import keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
from horovod.keras import DistributedOptimizer


class NormOut(Layer):
    def __init__(self, **kwargs):
        self.mean_dense = Dense(1, **kwargs)
        self.sd_dense = Dense(1, activation=K.tf.exp, **kwargs)
        super(NormOut, self).__init__()

    def call(self, inputs, **kwargs):
        mean_x = self.mean_dense(inputs)
        sd_x = self.sd_dense(inputs)
        return Concatenate()([mean_x, sd_x])

    def compute_output_shape(self, input_shape):
        return input_shape[0], 2


class GaussianMixtureOut(Layer):
    def __init__(self, mixtures=2, **kwargs):
        self.mixtures = mixtures
        self.mean_dense = Dense(self.mixtures, activation="relu", **kwargs)
        self.sd_dense = Dense(self.mixtures, activation=K.tf.exp, **kwargs)
        self.weight_dense = Dense(self.mixtures, activation="softmax", **kwargs)
        super(GaussianMixtureOut, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mean_x = self.mean_dense(inputs)
        sd_x = self.sd_dense(inputs)
        weights_x = self.weight_dense(inputs)
        return Concatenate()([mean_x, sd_x, weights_x])

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.mixtures * 3


def crps_norm(y_true, y_pred, cdf_points=np.arange(-200, 200.0, 1.0)):
    cdf_points_tensor = K.tf.constant(0.5 * (cdf_points[:-1] + cdf_points[1:]), dtype="float32")
    cdf_point_diffs = K.tf.constant(cdf_points[1:] - cdf_points[:-1], dtype="float32")
    y_pred_cdf = tfd.Normal(loc=y_pred[:, 0:1], scale=y_pred[:, 1:2]).cdf(cdf_points_tensor)
    y_true_cdf = K.tf.cast(y_true <= cdf_points_tensor, "float32")
    cdf_diffs = K.tf.reduce_mean((y_pred_cdf - y_true_cdf) ** 2 * cdf_point_diffs, axis=1)
    return K.tf.reduce_mean(cdf_diffs)


def crps_mixture(y_true, y_pred, cdf_points=np.arange(0, 200.0, 5.0)):
    cdf_points_tensor = K.tf.constant(0.5 * (cdf_points[:-1] + cdf_points[1:]), dtype="float32")
    cdf_point_diffs = K.tf.constant(cdf_points[1:] - cdf_points[:-1], dtype="float32")
    num_mixtures = y_pred.shape[1] // 3
    weights = [y_pred[:, 2 * num_mixtures + i: 2 * num_mixtures + i + 1] for i in range(num_mixtures)]
    locs = [y_pred[:, i:i+1] for i in range(num_mixtures)]
    scales = [y_pred[:, num_mixtures + i: num_mixtures + i + 1] for i in range(num_mixtures)]
    y_pred_cdf = K.tf.add_n([weights[i] * tfd.Normal(loc=locs[i], scale=scales[i]).cdf(cdf_points_tensor)
                             for i in range(num_mixtures)])
    y_true_cdf = K.tf.cast(y_true <= cdf_points_tensor, "float32")
    cdf_diffs = K.tf.reduce_mean((y_pred_cdf - y_true_cdf) ** 2 * cdf_point_diffs, axis=1)
    return K.tf.reduce_mean(cdf_diffs)


losses = {"mse": mean_squared_error,
          "crps_norm": crps_norm,
          "crps_mixture": crps_mixture}


class DenseNeuralNet(object):
    def __init__(self, hidden_layers=1, hidden_neurons=10, activation="relu", learning_rate=0.001,
                 output_type="linear", optimizer="adam", dropout_alpha=0.0, batch_size=64, epochs=10,
                 verbose=0, metrics=None, leaky_alpha=0.1, distributed=False):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = activation
        self.learning_rate = learning_rate
        self.output_type = output_type
        self.num_mixtures = 1
        if output_type == "gaussian":
            self.loss = losses["crps_norm"]
        elif "mixture" in output_type:
            self.num_mixtures = int(output_type.split("_")[1])
            self.loss = losses["crps_mixture"]
        else:
            self.loss = losses["mse"]
        self.dropout_alpha = dropout_alpha
        self.optimizer = optimizer
        self.epochs = epochs
        self.leaky_alpha = leaky_alpha
        self.use_dropout = False
        if self.dropout_alpha > 0:
            self.use_dropout = True
        self.batch_size = batch_size
        self.verbose = verbose
        self.metrics = metrics
        self.model = None
        self.parallel_model = None
        self.distributed = distributed

    def build_network(self, scalar_input_shape, output_size):
        ann_input = Input(shape=(scalar_input_shape,))
        ann_model = ann_input
        for h in range(self.hidden_layers):
            if self.use_dropout:
                ann_model = Dropout(self.dropout_alpha)(ann_model)
            ann_model = Dense(self.hidden_neurons)(ann_model)
            if self.hidden_activation == "leaky":
                ann_model = LeakyReLU(self.leaky_alpha, name=f"hidden_scalar_activation_{h:02d}")(ann_model)
            else:
                ann_model = Activation(self.hidden_activation, name=f"hidden_scalar_activation_{h:02d}")(ann_model)
        if self.output_type == "gaussian":
            ann_model = NormOut()(ann_model)
        elif self.output_type.split("_")[0] == "mixture":
            num_mixtures = int(self.output_type.split("_")[1])
            ann_model = GaussianMixtureOut(mixtures=num_mixtures)(ann_model)
        else:
            ann_model = Dense(output_size)(ann_model)
        self.model = Model(ann_input, ann_model)
        print(self.model.summary())
        return

    def compile_model(self):
        """
        Compile the model in tensorflow with the right optimizer and loss function.
        """
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        if self.distributed:
            opt = DistributedOptimizer(opt)
        self.model.compile(opt, self.loss, metrics=self.metrics)

    def compile_parallel_model(self, num_gpus):
        self.parallel_model = multi_gpu_model(self.model, num_gpus)
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)

        self.parallel_model.compile(opt, self.loss, metrics=self.metrics)

    @staticmethod
    def get_data_shapes(x, y):
        if len(y.shape) == 2:
            y_shape = y.shape[1]
        else:
            y_shape = 1
        return x.shape[1], y_shape

    def fit(self, x, y, val_x=None, val_y=None, build=True, **kwargs):
        if build:
            x_scalar_shape, y_size = self.get_data_shapes(x, y)
            self.build_network(x_scalar_shape, y_size)
            self.compile_model()
        if val_x is None:
            val_data = None
        else:
            val_data = (val_x, val_y)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       validation_data=val_data,  **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)


class BaseConvNet(object):
    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=5, min_data_width=4, pooling_width=2,
                 hidden_activation="relu", output_type="linear",
                 pooling="mean", use_dropout=False, dropout_alpha=0.0,
                 data_format="channels_first", optimizer="adam", loss="mse", leaky_alpha=0.1, metrics=None,
                 learning_rate=0.001, batch_size=1024, epochs=10, verbose=0, l2_alpha=0, distributed=False):
        self.min_filters = min_filters
        self.filter_width = filter_width
        self.filter_growth_rate = filter_growth_rate
        self.pooling_width = pooling_width
        self.min_data_width = min_data_width
        self.hidden_activation = hidden_activation
        self.output_type = output_type
        self.use_dropout = use_dropout
        self.pooling = pooling
        self.dropout_alpha = dropout_alpha
        self.data_format = data_format
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.leaky_alpha = leaky_alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.parallel_model = None
        self.l2_alpha = l2_alpha
        if l2_alpha > 0:
            self.use_l2 = True
        else:
            self.use_l2 = False
        self.verbose = verbose
        self.distributed = distributed

    def build_network(self, conv_input_shape, output_size):
        """
        Create a keras model with the hyperparameters specified in the constructor.

        Args:
            conv_input_shape (tuple of shape [variable, y, x]): The shape of the input data
            output_size: Number of neurons in output layer.
        """
        print("Conv input shape", conv_input_shape)
        if self.use_l2:
            reg = l2(self.l2_alpha)
        else:
            reg = None
        conv_input_layer = Input(shape=conv_input_shape, name="conv_input")
        num_conv_layers = int(np.round((np.log(conv_input_shape[1]) - np.log(self.min_data_width))
                                       / np.log(self.pooling_width)))
        print(num_conv_layers)
        num_filters = self.min_filters
        scn_model = conv_input_layer
        for c in range(num_conv_layers):
            scn_model = Conv2D(num_filters, (self.filter_width, self.filter_width),
                               data_format=self.data_format,
                               kernel_regularizer=reg, padding="same", name="conv_{0:02d}".format(c))(scn_model)
            if self.hidden_activation == "leaky":
                scn_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(c))(scn_model)
            else:
                scn_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(c))(scn_model)
            if self.use_dropout:
                scn_model = SpatialDropout2D(rate=self.dropout_alpha)(scn_model)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                scn_model = MaxPool2D(pool_size=(self.pooling_width, self.pooling_width),
                                      data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
            else:
                scn_model = AveragePooling2D(pool_size=(self.pooling_width, self.pooling_width),
                                             data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
        scn_model = Flatten(name="flatten")(scn_model)

        if self.output_type == "linear":
            scn_model = Dense(output_size, name="dense_output")(scn_model)
            scn_model = Activation("linear", name="activation_output")(scn_model)
        elif self.output_type == "gaussian":
            scn_model = NormOut()(scn_model)
        elif "mixture" in self.output_type:
            num_mixtures = int(self.output_type.split("_")[1])
            scn_model = GaussianMixtureOut(mixtures=num_mixtures)(scn_model)
        self.model = Model(conv_input_layer, scn_model)
        print(self.model.summary())

    def compile_model(self):
        """
        Compile the model in tensorflow with the right optimizer and loss function.
        """
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        if self.distributed:
            opt = DistributedOptimizer(opt)
        self.model.compile(opt, losses[self.loss], metrics=self.metrics)

    def compile_parallel_model(self, num_gpus):
        self.parallel_model = multi_gpu_model(self.model, num_gpus)
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        self.parallel_model.compile(opt, losses[self.loss], metrics=self.metrics)

    @staticmethod
    def get_data_shapes(x, y):
        """
        Extract the input and output data shapes in order to construct the neural network.
        """
        if len(x.shape) != 4:
            raise ValueError("Input data does not have dimensions (examples, y, x, predictor)")
        if len(y.shape) == 1:
            output_size = 1
        else:
            output_size = y.shape[1]
        return x.shape[1:], output_size

    @staticmethod
    def get_generator_data_shapes(data_gen):
        inputs, outputs = data_gen.__getitem__(0)
        if len(outputs.shape) == 1:
            output_size = 1
        else:
            output_size = outputs.shape[1]
        return inputs.shape[1:], output_size

    def fit(self, x, y, val_x=None, val_y=None, build=True, **kwargs):
        """
        Train the neural network.
        """
        if build:
            x_conv_shape, y_size = self.get_data_shapes(x, y)
            self.build_network(x_conv_shape, y_size)
            self.compile_model()
        if val_x is None:
            val_data = None
        else:
            val_data = (val_x, val_y)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       validation_data=val_data, **kwargs)

    def fit_generator(self, generator, build=True, validation_generator=None, **kwargs):
        if build:
            x_conv_shape, y_size = self.get_generator_data_shapes(generator)
            self.build_network(x_conv_shape, y_size)
            self.compile_model()
        self.model.fit_generator(generator, epochs=self.epochs, verbose=self.verbose,
                                 validation_data=validation_generator, **kwargs)

    def predict(self, x, y):
        return self.model.predict(x, y, batch_size=self.batch_size)

class StandardConvNet(object):
    """
    Standard Convolutional Neural Network contains a series of convolution and pooling layers followed by one
    fully connected layer to a set of scalar outputs. The number of convolution filters is assumed to increase
    with depth.

    Attributes:
        min_filters (int): The number of convolution filters in the first layer.
        filter_growth_rate (float): Multiplier on the number of convolution filters between layers.
        filter_width (int): Width in number of pixels in each convolution filter
        min_data_width (int): The minimum dimension of the input data after the final pooling layer. Constrains the number of
            convolutional layers.
        pooling_width (int): Width of pooling layer
        scalar_hidden_layers (int): Number of dense hidden layers for transforming the scalar values.
        scalar_hidden_neurons (int): Number of neurons in each dense hidden layer.
        hidden_activation (str): The nonlinear activation function applied after each convolutional layer. If "leaky", a leaky ReLU with
            alpha=0.1 is used.
        output_type (str): "linear" (deterministic), "gaussian" (mean and standard deviation), or "mixture_2"
            (gaussian mixture model with number of Gaussians as the part the underscore.
        pooling (str): If mean, then :class:`keras.layers.AveragePooling2D` is used for pooling. If max, then :class:`keras.layers.MaxPool2D` is used.
        use_dropout (bool): If True, then a :class:`keras.layers.Dropout` layer is inserted between the final convolution block
            and the output :class:`keras.laysers.Dense` layer.
        dropout_alpha (float): probability of a neuron being set to zero.
        data_format (str): "channels_first" or "channels_last"
        optimizer (str): Name of the optimizer being used. "sgd" and "adam" are supported.
        loss (str): "mse", "crps_norm", or "crps_mixture"
        leaky_alpha (float): Scaling factor for leaky ReLU activation.
        metrics (None or list): Metrics to track during training
        learning_rate (float): optimization learning rate
        batch_size (int): Number of samples per batch
        epochs (int): Number of passes through training data
        verbose (int): Level of verbosity
    """

    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=5, min_data_width=4, pooling_width=2,
                 scalar_hidden_layers=1, scalar_hidden_neurons=30,
                 hidden_activation="relu", output_type="linear",
                 pooling="mean", use_dropout=False, dropout_alpha=0.0,
                 data_format="channels_first", optimizer="adam", loss="mse", leaky_alpha=0.1, metrics=None,
                 learning_rate=0.001, batch_size=1024, epochs=10, verbose=0, l2_alpha=0, distributed=False):
        self.min_filters = min_filters
        self.filter_width = filter_width
        self.filter_growth_rate = filter_growth_rate
        self.pooling_width = pooling_width
        self.min_data_width = min_data_width
        self.scalar_hidden_layers = scalar_hidden_layers
        self.scalar_hidden_neurons = scalar_hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_type = output_type
        self.use_dropout = use_dropout
        self.pooling = pooling
        self.dropout_alpha = dropout_alpha
        self.data_format = data_format
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.leaky_alpha = leaky_alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.parallel_model = None
        self.l2_alpha = l2_alpha
        if l2_alpha > 0:
            self.use_l2 = True
        else:
            self.use_l2 = False
        self.verbose = verbose
        self.distributed = distributed


    def build_network(self, scalar_input_shape, conv_input_shape, output_size):
        """
        Create a keras model with the hyperparameters specified in the constructor.

        Args:
            scalar_input_shape: Number of columns in input table
            conv_input_shape (tuple of shape [variable, y, x]): The shape of the input data
            output_size: Number of neurons in output layer.
        """
        print("Scalar input shape", scalar_input_shape)
        print("Conv input shape", conv_input_shape)
        if self.use_l2:
            reg = l2(self.l2_alpha)
        else:
            reg = None
        scalar_input_layer = Input(shape=(scalar_input_shape,), name="scalar_input")
        conv_input_layer = Input(shape=conv_input_shape, name="conv_input")
        num_conv_layers = int(np.round((np.log(conv_input_shape[1]) - np.log(self.min_data_width))
                                       / np.log(self.pooling_width)))
        print(num_conv_layers)
        num_filters = self.min_filters
        scn_model = conv_input_layer
        for c in range(num_conv_layers):
            scn_model = Conv2D(num_filters, (self.filter_width, self.filter_width),
                               data_format=self.data_format, 
                               kernel_regularizer=reg, padding="same", name="conv_{0:02d}".format(c))(scn_model)
            if self.hidden_activation == "leaky":
                scn_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(c))(scn_model)
            else:
                scn_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(c))(scn_model)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                scn_model = MaxPool2D(pool_size=(self.pooling_width, self.pooling_width),
                                      data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
            else:
                scn_model = AveragePooling2D(pool_size=(self.pooling_width, self.pooling_width),
                                             data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
        scn_model = Flatten(name="flatten")(scn_model)
        scalar_model = scalar_input_layer
        for h in range(self.scalar_hidden_layers):
            scalar_model = Dense(self.scalar_hidden_neurons, kernel_regularizer=reg, name=f"scalar_dense_{h:02d}")(scalar_model)
            if self.hidden_activation == "leaky":
                scalar_model = LeakyReLU(self.leaky_alpha, name=f"hidden_scalar_activation_{h:02d}")(scalar_model)
            else:
                scalar_model = Activation(self.hidden_activation, name=f"hidden_scalar_activation_{h:02d}")(scalar_model)
        scn_model = Concatenate()([scalar_model, scn_model])
        if self.use_dropout:
            scn_model = Dropout(self.dropout_alpha, name="dense_dropout")(scn_model)
        if self.output_type == "linear":
            scn_model = Dense(output_size, name="dense_output")(scn_model)
            scn_model = Activation("linear", name="activation_output")(scn_model)
        elif self.output_type == "gaussian":
            scn_model = NormOut()(scn_model)
        elif "mixture" in self.output_type:
            num_mixtures = int(self.output_type.split("_")[1])
            scn_model = GaussianMixtureOut(mixtures=num_mixtures)(scn_model)
        self.model = Model([scalar_input_layer, conv_input_layer], scn_model)
        print(self.model.summary())

    def compile_model(self):
        """
        Compile the model in tensorflow with the right optimizer and loss function.
        """
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        if self.distributed:
            opt = DistributedOptimizer(opt)
        self.model.compile(opt, losses[self.loss], metrics=self.metrics)

    def compile_parallel_model(self, num_gpus):
        self.parallel_model = multi_gpu_model(self.model, num_gpus)
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        self.parallel_model.compile(opt, losses[self.loss], metrics=self.metrics)

    @staticmethod
    def get_data_shapes(x, y):
        """
        Extract the input and output data shapes in order to construct the neural network.
        """
        if len(x[1].shape) != 4:
            raise ValueError("Input data does not have dimensions (examples, y, x, predictor)")
        if len(y.shape) == 1:
            output_size = 1
        else:
            output_size = y.shape[1]
        return x[0].shape[1], x[1].shape[1:], output_size

    @staticmethod
    def get_generator_data_shapes(data_gen):
        inputs, outputs = data_gen.__getitem__(0)
        if len(outputs.shape) == 1:
            output_size = 1
        else:
            output_size = outputs.shape[1]
        return inputs[0].shape[1], inputs[1].shape[1:], output_size

    def fit(self, x, y, val_x=None, val_y=None, build=True, **kwargs):
        """
        Train the neural network.
        """
        if build:
            x_scalar_shape, x_conv_shape, y_size = self.get_data_shapes(x, y)
            self.build_network(x_scalar_shape, x_conv_shape, y_size)
            self.compile_model()
        if val_x is None:
            val_data = None
        else:
            val_data = (val_x, val_y)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       validation_data=val_data, **kwargs)

    def fit_generator(self, generator, build=True, validation_generator=None, **kwargs):
        if build:
            x_scalar_shape, x_conv_shape, y_size = self.get_generator_data_shapes(generator)
            self.build_network(x_scalar_shape, x_conv_shape, y_size)
            self.compile_model()
        self.model.fit_generator(generator, epochs=self.epochs, verbose=self.verbose,
                                 validation_data=validation_generator, **kwargs)

    def predict(self, x, y):
        return self.model.predict(x, y, batch_size=self.batch_size)


class ResNet(BaseConvNet):
    """
    Extension of the :class:`marbleri.models.StandardConvNet` to include Residual layers instead of single convolutional layers.
    The residual layers split the data signal off, apply normalization and convolutions to it, then adds it back on to the original field.
    """

    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=5, min_data_width=4, pooling_width=2,
                 hidden_activation="relu", output_type="linear",
                 pooling="mean", use_dropout=False, dropout_alpha=0.0,
                 data_format="channels_first", optimizer="adam", loss="mse", leaky_alpha=0.1, metrics=None,
                 learning_rate=0.001, batch_size=1024, epochs=10, l2_alpha=0, verbose=0, distributed=False, **kwargs):
        super().__init__(min_filters=min_filters, filter_growth_rate=filter_growth_rate, filter_width=filter_width,
                         min_data_width=min_data_width, pooling_width=pooling_width,
                         hidden_activation=hidden_activation, data_format=data_format,
                         output_type=output_type, pooling=pooling, use_dropout=use_dropout,
                         dropout_alpha=dropout_alpha, optimizer=optimizer, loss=loss, metrics=metrics,
                         leaky_alpha=leaky_alpha,
                         batch_size=batch_size, epochs=epochs, verbose=verbose, learning_rate=learning_rate, l2_alpha=l2_alpha)

    def residual_block(self, filters, in_layer, layer_number=0):
        """
        Generate a single residual block.
        """
        if self.data_format == "channels_first":
            norm_axis = 1
        else:
            norm_axis = -1
        if in_layer.shape[norm_axis].value != filters:
            x = Conv2D(filters, self.filter_width, data_format=self.data_format, padding="same")(in_layer)
        else:
            x = in_layer
        y = BatchNormalization(axis=norm_axis, name="bn_res_{0:02d}_a".format(layer_number))(x)
        if self.hidden_activation == "leaky":
            y = LeakyReLU(self.leaky_alpha, name="res_activation_{0:02d}_a".format(layer_number))(y)
        else:
            y = Activation(self.hidden_activation,
                           name="res_activation_{0:02d}_a".format(layer_number))(y)
        y = Conv2D(filters, self.filter_width, padding="same",
                   data_format=self.data_format, name="res_conv_{0:02d}_a".format(layer_number))(y)
        y = BatchNormalization(axis=norm_axis, name="bn_res_{0:02d}_b".format(layer_number))(y)
        if self.hidden_activation == "leaky":
            y = LeakyReLU(self.leaky_alpha, name="res_activation_{0:02d}_b".format(layer_number))(y)
        else:
            y = Activation(self.hidden_activation,
                           name="res_activation_{0:02d}_b".format(layer_number))(y)
        y = Conv2D(filters, self.filter_width, padding="same",
                   data_format=self.data_format, name="res_conv_{0:02d}_b".format(layer_number))(y)
        out = Add()([y, x])
        return out

    def build_network(self, conv_input_shape, output_size):
        print(conv_input_shape)
        conv_input_layer = Input(shape=conv_input_shape, name="conv_input")
        num_conv_layers = int(np.round((np.log(conv_input_shape[1]) - np.log(self.min_data_width))
                                       / np.log(self.pooling_width)))
        num_filters = self.min_filters
        res_model = conv_input_layer
        for c in range(num_conv_layers):
            res_model = self.residual_block(num_filters, res_model, c)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                res_model = MaxPool2D(data_format=self.data_format, name="pooling_{0:02d}".format(c))(res_model)
            else:
                res_model = AveragePooling2D(data_format=self.data_format, name="pooling_{0:02d}".format(c))(res_model)
        res_model = Flatten(name="flatten")(res_model)
        if self.use_dropout:
            res_model = Dropout(self.dropout_alpha, name="dense_dropout")(res_model)
        if self.output_type == "linear":
            res_model = Dense(output_size, name="dense_output")(res_model)
            res_model = Activation("linear", name="activation_output")(res_model)
        elif self.output_type == "gaussian":
            res_model = NormOut()(res_model)
        elif "mixture" in self.output_type:
            num_mixtures = int(self.output_type.split("_")[1])
            res_model = GaussianMixtureOut(mixtures=num_mixtures)(res_model)
        self.model = Model(conv_input_layer, res_model)
