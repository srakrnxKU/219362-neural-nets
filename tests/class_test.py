import nn
import numpy as np

class TestNN:
    nn_1 = nn.NN([3, 4, 5])
    def test_nn_layers(self):
        self.nn_1.weights[0].shape = [3, 4]
        self.nn_1.weights[1].shape = [4, 5]