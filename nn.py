import numpy as np
import itertools


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


logistic = lambda x: 1 / (1 + np.e ** (-x))
logistic_derivative: lambda x: logistic(x) * (1 - logistic(x))


class NN:
    def __init__(self, layers, random_state=None):
        self.rng = np.random.RandomState(random_state)
        if len(list(layers)) < 2:
            raise ValueError("Invalid network size.")
        self.weights = [None] * (len(layers) - 1)
        for i, size in enumerate(pairwise(layers)):
            self.weights[i] = self.rng.rand(*size)

    def forward(self, values):
        values = np.array(values)
        for weight in self.weights:
            values = np.matmul(values, weight)
            values = logistic(values)
        return values


def main():
    pass


if __name__ == "__main__":
    main()
