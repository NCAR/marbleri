from .models import StandardConvNet, crps_norm, crps_mixture, ResNet, NormOut, GaussianMixtureOut
import keras.backend as K
import numpy as np


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

    print(crps_val)


def test_StandardConvNet():
    scalar_input_shape = 10
    conv_input_shape = (12, 27, 27)
    scn_norm = StandardConvNet(min_filters=16, output_type="gaussian", data_format="channels_first",
                          loss="crps_norm", pooling_width=3)
    scn_norm.build_network(scalar_input_shape, conv_input_shape, 1)
    scn_norm.compile_model()
    print(scn_norm.model.summary())
    assert scn_norm.model.input[0].get_shape()[1] == scalar_input_shape
    assert scn_norm.model.input[1].get_shape()[1] == conv_input_shape[0]
    assert scn_norm.model.output.get_shape()[1] == 2


def test_ResNet():
    conv_input_shape = (9, 48, 48)
    num_examples = 16
    rn = ResNet(min_filters=16, filter_growth_rate=1.5, min_data_width=6, filter_width=3, epochs=1,
                hidden_activation="leaky", data_format="channels_first", pooling_width=2,
                output_type='linear', loss="mse", pooling="max", learning_rate=0.0001, verbose=1)
    x_data = np.random.normal(size=[num_examples] + list(conv_input_shape))
    y_data = np.random.normal(size=num_examples)
    assert len(x_data.shape) == 4
    rn.fit(x_data, y_data)
    print(rn.model.summary())
    config = {'min_filters': 16, 'min_data_width': 6,
              'filter_width': 3, 'filter_growth_rate': 1.5,
              'pooling_width': 2, 'hidden_activation': 'leaky',
              'output_type': 'linear', 'loss': 'mse',
              'pooling': 'max',
              'data_format': 'channels_first',
              'optimizer': 'adam', 'batch_size': 16,
              'epochs': 1, 'learning_rate': 0.0001, 'verbose': 1}
    rn2 = ResNet(**config)
    rn2.fit(x_data, y_data)
    rn2.model.summary()
    return