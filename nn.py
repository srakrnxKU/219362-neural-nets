import numpy as np
import itertools

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

class NN:
    def __init__(self, layers):
        self.weights = [None] * 3
        for i, size in enumerate(pairwise(layers)):
            self.weights[i] = np.zeros(size)

def main():
    pass

if __name__ == "__main__":
    main()