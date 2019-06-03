from .models import StandardConvNet, crps_norm, crps_mixture, ResNet, NormOut, GaussianMixtureOut
import keras.backend as K
import numpy as np


def test_NormOut():
    input = K.placeholder(shape=(None, 5))
    nout = NormOut()
    norm_func = K.function([input], [nout(input)])
    result = norm_func([np.zeros((3, 5))])[0]
    assert result.shape[0] == 3
    assert result.shape[1] == 2


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
