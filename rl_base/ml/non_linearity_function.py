import abc
from typing import Type, Union


class NonLinearityFunction(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calc(x):
        pass

    @staticmethod
    @abc.abstractmethod
    def deriv(x):
        pass


NonLinearityFunctionType = Union[Type[NonLinearityFunction], NonLinearityFunction]
