import numpy as np
import itertools

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def relu(x):
    return x if x > 0 else 0

def relu_derivative(x):
    return 1 if x > 0 else 0

class NN:
    def __init__(self, layers, random_state=None):
        self.rng = np.random.RandomState(random_state)
        if len(list(layers)) < 2:
            raise ValueError("Invalid network size.")
        self.weights = [None] * 3
        for i, size in enumerate(pairwise(layers)):
            self.weights[i] = self.rng.rand(*size)

def main():
    pass

if __name__ == "__main__":
    main()