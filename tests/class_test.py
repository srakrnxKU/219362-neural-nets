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

    def test_feed_forward(self):
        x = np.array([[1, 1, 1], [2, 2, 2]])
        output = self.nn_1.forward(x)
        assert output.shape == (2, 5)


class TestNNBehaviour:
    nn_1 = nn.NN([3, 4, 5], random_state=1)
    nn_2 = nn.NN([3, 4, 5], random_state=1)
    nn_3 = nn.NN([3, 4, 5], random_state=3)

    def test_random_state(self):
        print(self.nn_1.weights[0])
        for i in range(2):
            np.testing.assert_equal(self.nn_1.weights[i], self.nn_2.weights[i])
        for i in range(2):
            with pytest.raises(AssertionError):
                np.testing.assert_equal(self.nn_1.weights[i], self.nn_3.weights[i])
