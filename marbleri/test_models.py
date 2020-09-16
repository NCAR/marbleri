from .models import MixedConvNet, crps_norm, ResNet, NormOut, BaseConvNet
import tensorflow.keras.backend as K
import numpy as np
import yaml


def test_NormOut():
    input = K.placeholder(shape=(None, 5))
    nout = NormOut()
    norm_func = K.function([input], [nout(input)])
    result = norm_func([np.ones((3, 5))])[0]
    print(result)
    assert result.shape[0] == 3
    assert result.shape[1] == 2


def test_crps_norm():
    y_true = np.random.randint(20, 160, size=(10, 1)).astype(np.float32)
    #y_pred_mean = np.array(y_true[:, 0])
    y_pred_mean = np.ones(y_true.shape[0]) * 5
    y_pred_sd = np.ones(y_true.size) * 5.0
    y_pred = np.vstack([y_pred_mean, y_pred_sd]).T
    y_pred_t = K.placeholder(shape=(None, 2))
    y_true_t = K.placeholder(shape=(None, 1))
    crps_func = K.function([y_true_t, y_pred_t], [crps_norm(y_true_t, y_pred_t)])
    assert y_pred.shape[0] == y_true.shape[0]
    crps_val = crps_func([y_true, y_pred])[0]
    assert crps_val >= 0
    assert ~np.isnan(crps_val)

def test_MixedConvNet():
    scalar_input_shape = 10
    conv_input_shape = (96, 96, 12)
    scn_norm = MixedConvNet(min_filters=16, output_type="gaussian", data_format="channels_last",
                          loss="crps_norm", pooling_width=2)
    scn_norm.build_network(scalar_input_shape, conv_input_shape, 1)
    scn_norm.compile_model()
    print(scn_norm.model_.summary())
    assert scn_norm.model_.input[0].get_shape()[1] == scalar_input_shape
    assert scn_norm.model_.input[1].get_shape()[1] == conv_input_shape[0]
    assert scn_norm.model_.output.get_shape()[1] == 2


def test_BaseConvNet():
    conv_input_shape = (96, 96, 12)
    config_str = \
      """config:
            min_filters: 16
            pooling_width: 2
            dense_neurons: 32
            min_data_width: 12
            output_type: "linear"
            loss: "mae"
            pooling: "max"
            data_format: "channels_last"
            l2_alpha: 0
            optimizer: "adam"
            batch_size: 32
            epochs: 3
            learning_rate: 0.01
            verbose: 1
    """
    num_x = 128
    x = np.random.random(size=(num_x, conv_input_shape[0], conv_input_shape[1], conv_input_shape[2]))
    config = yaml.load(config_str, Loader=yaml.Loader)
    bcn = BaseConvNet(**config["config"])
    bcn.build_network(conv_input_shape, 1)
    bcn.compile_model()
    bcn.fit(x, x[:, conv_input_shape[0] // 2, conv_input_shape[0] // 2, 0])
    print(bcn.model_.summary())
    assert bcn.model_ is not None
    return


def test_ResNet():
    conv_input_shape = (48, 48, 9)
    num_examples = 16
    rn = ResNet(min_filters=16, filter_growth_rate=1.5, min_data_width=6, filter_width=3, epochs=1,
                hidden_activation="leaky", data_format="channels_last", pooling_width=2,
                output_type='linear', loss="mae", pooling="max", learning_rate=0.0001, verbose=1)
    x_data = np.random.normal(size=[num_examples] + list(conv_input_shape))
    y_data = np.random.normal(size=num_examples)
    assert len(x_data.shape) == 4
    rn.fit(x_data, y_data)
    print(rn.model_.summary())
    config = {'min_filters': 16, 'min_data_width': 6,
              'filter_width': 3, 'filter_growth_rate': 1.5,
              'pooling_width': 2, 'hidden_activation': 'leaky',
              'output_type': 'linear', 'loss': 'mse',
              'pooling': 'max',
              'data_format': 'channels_last',
              'optimizer': 'adam', 'batch_size': 16,
              'epochs': 1, 'learning_rate': 0.0001, 'verbose': 1}
    rn2 = ResNet(**config)
    rn2.fit(x_data, y_data)
    rn2.model_.summary()
    return

