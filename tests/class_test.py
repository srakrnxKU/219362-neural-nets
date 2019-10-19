import nn
import numpy as np
import pytest


class TestNN:
    nn_1 = nn.NN([3, 4, 5])

    def test_nn_layers(self):
        assert self.nn_1.weights[0].shape == (3, 4)
        assert self.nn_1.weights[1].shape == (4, 5)

    def test_invalid_nn_layers(self):
        with pytest.raises(ValueError):
            assert nn.NN([])
        with pytest.raises(ValueError):
            assert nn.NN([1])
