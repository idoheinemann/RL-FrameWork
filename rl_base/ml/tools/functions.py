import numpy as np

from rl_base.ml.non_linearity_function import NonLinearityFunction


class Sigmoid(NonLinearityFunction):
    @staticmethod
    def calc(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def deriv(x):
        return x * (1 - x)


class Tanh(NonLinearityFunction):
    @staticmethod
    def calc(x):
        return np.tanh(x)

    @staticmethod
    def deriv(x):
        return 1 - x * x


class Relu(NonLinearityFunction):
    @staticmethod
    def calc(x):
        e = np.exp(x)
        return np.log(e + 1)

    @staticmethod
    def deriv(x):
        e = np.exp(x)
        return (e - 1) / e


class LinearRelu(NonLinearityFunction):
    @staticmethod
    @np.vectorize
    def calc(x):
        return x if x > 0.0 else 0.0

    @staticmethod
    def deriv(x):
        return np.ones_like(x)


class Linear(NonLinearityFunction):
    @staticmethod
    def calc(x):
        return x

    @staticmethod
    def deriv(x):
        return 1


class ArcTan(NonLinearityFunction):
    @staticmethod
    def calc(x):
        return np.arctan(x)

    @staticmethod
    def deriv(x):
        return 1 / (1 + np.tan(x) ** 2)


class SoftMax(Sigmoid):
    @staticmethod
    def calc(x):
        e = Sigmoid.calc(x)
        return e / np.sum(e)
