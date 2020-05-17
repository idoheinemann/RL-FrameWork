import abc
from typing import TypeVar

Data = TypeVar('Data')
Label = TypeVar('Label')


class PredictionModel(abc.ABC):
    @abc.abstractmethod
    def predict(self, data: Data) -> Label:
        pass

    @abc.abstractmethod
    def train(self, data: Data, label: Label) -> float:
        pass
