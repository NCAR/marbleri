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