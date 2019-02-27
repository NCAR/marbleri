from keras.layers import Dense, Conv2D, Activation, Input, Flatten, AveragePooling2D, MaxPool2D, LeakyReLU, Dropout, Add
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
import numpy as np
import keras.backend as K


class StandardConvNet(object):
    """
    Standard Convolutional Neural Network contains a series of convolution and pooling layers followed by one
    fully connected layer to a set of scalar outputs. The number of convolution filters is assumed to increase
    with depth.

    Attributes:
        min_filters (int): The number of convolution filters in the first layer.
        filter_growth_rate (float): Multiplier on the number of convolution filters between layers.
        min_data_width (int): The minimum dimension of the input data after the final pooling layer. Constrains the number of
            convolutional layers.
        hidden_activation (str): The nonlinear activation function applied after each convolutional layer. If "leaky", a leaky ReLU with
            alpha=0.1 is used.
        output_activation (str): The nonlinear activation function applied on the output layer.
        pooling (str): If mean, then :class:`keras.layers.AveragePooling2D` is used for pooling. If max, then :class:`keras.layers.MaxPool2D` is used.
        use_dropout (bool): If True, then a :class:`keras.layers.Dropout` layer is inserted between the final convolution block
            and the output :class:`keras.laysers.Dense` layer.

    """

    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=5, min_data_width=4,
                 hidden_activation="relu", output_activation="sigmoid",
                 pooling="mean", use_dropout=False, dropout_alpha=0.0,
                 optimizer="adam", loss="mse", leaky_alpha=0.1,
                 learning_rate=0.00001, batch_size=256, epochs=10, verbose=0):
        self.min_filters = min_filters
        self.filter_width = filter_width
        self.filter_growth_rate = filter_growth_rate
        self.min_data_width = min_data_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_dropout = use_dropout
        self.pooling = pooling
        self.dropout_alpha = dropout_alpha
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.leaky_alpha = leaky_alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.verbose = verbose

    def build_network(self, input_shape, output_size):
        """
        Create a keras model with the hyperparameters specified in the constructor.

        Args:
            input_shape (tuple of shape [y, x, variables]): The shape of the input data
            output_size: Number of neurons in output layer.
        """
        input_layer = Input(shape=input_shape, name="scn_input")
        num_conv_layers = int(np.log2(input_shape[1]) - np.log2(self.min_data_width))
        num_filters = self.min_filters
        scn_model = input_layer
        for c in range(num_conv_layers):
            scn_model = Conv2D(num_filters, (self.filter_width, self.filter_width),
                               padding="same", name="conv_{0:02d}".format(c))(scn_model)
            if self.hidden_activation == "leaky":
                scn_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(c))(scn_model)
            else:
                scn_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(c))(scn_model)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                scn_model = MaxPool2D(name="pooling_{0:02d}".format(c))(scn_model)
            else:
                scn_model = AveragePooling2D(name="pooling_{0:02d}".format(c))(scn_model)
        scn_model = Flatten(name="flatten")(scn_model)
        if self.use_dropout:
            scn_model = Dropout(self.dropout_alpha, name="dense_dropout")(scn_model)
        scn_model = Dense(output_size, name="dense_output")(scn_model)
        scn_model = Activation(self.output_activation, name="activation_output")(scn_model)
        self.model = Model(input_layer, scn_model)

    def compile_model(self):
        """
        Compile the model in tensorflow with the right optimizer and loss function.
        """
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        self.model.compile(opt, self.loss)

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

    def fit(self, x, y, val_x=None, val_y=None, build=True):
        """
        Train the neural network.
        """
        if build:
            x_shape, y_size = self.get_data_shapes(x, y)
            self.build_network(x_shape, y_size)
            self.compile_model()
        if val_x is None:
            val_data = None
        else:
            val_data = (val_x, val_y)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       validation_data=val_data)

    def predict(self, x, y):
        return self.model.predict(x, y, batch_size=self.batch_size)


class ResNet(StandardConvNet):
    """
    Extension of the :class:`goes16ci.models.StandardConvNet` to include Residual layers instead of single convolutional layers.
    The residual layers split the data signal off, apply normalization and convolutions to it, then adds it back on to the original field.
    """

    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=5, min_data_width=4,
                 hidden_activation="relu", output_activation="sigmoid",
                 pooling="mean", use_dropout=False, dropout_alpha=0.0, learning_rate=0.00001,
                 optimizer="adam", loss="mse", leaky_alpha=0.1, batch_size=256, epochs=10, verbose=0):
        super().__init__(min_filters=min_filters, filter_growth_rate=filter_growth_rate, filter_width=filter_width,
                         min_data_width=min_data_width, hidden_activation=hidden_activation,
                         output_activation=output_activation, pooling=pooling, use_dropout=use_dropout,
                         dropout_alpha=dropout_alpha, optimizer=optimizer, loss=loss, leaky_alpha=leaky_alpha,
                         batch_size=batch_size, epochs=epochs, verbose=verbose, learning_rate=learning_rate)

    def residual_block(self, filters, in_layer, layer_number=0):
        """
        Generate a single residual block.
        """
        if in_layer.shape[-1].value != filters:
            x = Conv2D(filters, self.filter_width, padding="same")(in_layer)
        else:
            x = in_layer
        y = BatchNormalization(name="bn_res_{0:02d}_a".format(layer_number))(x)
        if self.hidden_activation == "leaky":
            y = LeakyReLU(self.leaky_alpha, name="res_activation_{0:02d}_a".format(layer_number))(y)
        else:
            y = Activation(self.hidden_activation,
                           name="res_activation_{0:02d}_a".format(layer_number))(y)
        y = Conv2D(filters, self.filter_width, padding="same",
                   name="res_conv_{0:02d}_a".format(layer_number))(y)
        y = BatchNormalization(name="bn_res_{0:02d}_b".format(layer_number))(y)
        if self.hidden_activation == "leaky":
            y = LeakyReLU(self.leaky_alpha, name="res_activation_{0:02d}_b".format(layer_number))(y)
        else:
            y = Activation(self.hidden_activation,
                           name="res_activation_{0:02d}_b".format(layer_number))(y)
        y = Conv2D(filters, self.filter_width, padding="same",
                   name="res_conv_{0:02d}_b".format(layer_number))(y)
        out = Add()([y, x])
        return out

    def build_network(self, input_shape, output_size):
        input_layer = Input(shape=input_shape, name="scn_input")
        num_conv_layers = int(np.log2(input_shape[1]) - np.log2(self.min_data_width))
        print(num_conv_layers)
        num_filters = self.min_filters
        res_model = input_layer
        for c in range(num_conv_layers):
            res_model = self.residual_block(num_filters, res_model, c)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                res_model = MaxPool2D(name="pooling_{0:02d}".format(c))(res_model)
            else:
                res_model = AveragePooling2D(name="pooling_{0:02d}".format(c))(res_model)
        res_model = Flatten(name="flatten")(res_model)
        if self.use_dropout:
            res_model = Dropout(self.dropout_alpha, name="dense_dropout")(res_model)
        res_model = Dense(output_size, name="dense_output")(res_model)
        res_model = Activation(self.output_activation, name="activation_output")(res_model)
        self.model = Model(input_layer, res_model)
