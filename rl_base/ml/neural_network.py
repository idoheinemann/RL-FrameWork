from typing import Union, Iterable, List

import numpy as np

from rl_base.ml.non_linearity_function import NonLinearityFunctionType
from rl_base.ml.prediction_model import PredictionModel
from rl_base.ml.tools.functions import Sigmoid, Tanh

def unit(x):
    return x

class NeuralNetwork(PredictionModel):
    def __init__(self, layers_num: Iterable[int],
                 functions: Union[NonLinearityFunctionType, Iterable[NonLinearityFunctionType]], n_min=-0.5, n_max=0.5,
                 alpha=0.01, gradient_method=None):
        self.layers = []
        self.biases = []
        layers = list(layers_num)
        if hasattr(functions, '__iter__') or hasattr(functions, '__getitem__'):
            functions = list(functions)
            assert len(layers) - 1 == len(functions)
        else:
            functions = [functions] * (len(layers) - 1)
        self.non_lin_funcs: List[NonLinearityFunctionType] = functions
        for i in range(len(layers) - 1):
            self.layers.append((n_max - n_min) * np.random.rand(layers[i], layers[i + 1]) + n_min)
            self.biases.append((n_max - n_min) * np.random.rand(layers[i + 1]) + n_min)
        if isinstance(alpha, float):
            alpha = [alpha] * len(self.layers)
        self.alpha = alpha
        if gradient_method is None:
            gradient_method = unit
        self.gradient_method = gradient_method

    def predict(self, data):
        for i in range(len(self.layers)):
            data = self.non_lin_funcs[i].calc(data.dot(self.layers[i]) + self.biases[i])
        return data

    def cost(self, data, label):
        return np.sum((self.predict(data) - label) ** 2)

    def train(self, data, label):
        inp = data
        inputs = []
        for i, layer in enumerate(self.layers):
            inputs.append(inp)
            inp = self.non_lin_funcs[i].calc(np.dot(inp, layer) + self.biases[i])

        error = label - inp
        deltas = [self.non_lin_funcs[-1].deriv(inp) * error]
        for i in range(len(self.layers) - 1, 0, -1):
            deltas.append(self.non_lin_funcs[i-1].deriv(inputs[i]) * np.dot(deltas[-1], self.layers[i].T))
            # delta[i] = (delta[i+1]*layer[i+1]) .* f[i]'(input[i])

        deltas.reverse()
        for i, delta in enumerate(deltas):
            self.layers[i] = self.layers[i] + self.alpha[i] * self.gradient_method(np.dot(inputs[i].T, delta))
            self.biases[i] = self.biases[i] + self.alpha[i] * self.gradient_method(np.sum(deltas[i], axis=0))
        return np.mean(error * error)


def main():
    nn = NeuralNetwork((2, 4, 1), Tanh, alpha=1.0)
    data = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]]).astype(float)
    label = np.array([[-1, 1, 1, -1]]).T.astype(float)
    print(nn.predict(data))
    for i in range(10000):
        nn.train(data, label)
    print(nn.predict(data))


if __name__ == '__main__':
    main()
