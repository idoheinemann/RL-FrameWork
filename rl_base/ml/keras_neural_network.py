from typing import Iterable, Union

import tensorflow as tf

from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.models import Sequential

from rl_base.ml.prediction_model import PredictionModel, Data, Label


class KerasNeuralNetwork(PredictionModel):
    def __init__(self, layers: Iterable[int], funcs: Union[str, Iterable[str]], batch_size=None, max_cores=8):
        session_conf = tf.compat.v1.ConfigProto(device_count={"CPU": max_cores})
        sess = tf.compat.v1.Session(config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)

        layers = list(layers)
        self.input_shape = layers.pop(0),
        if isinstance(funcs, str):
            funcs = [funcs] * len(layers)

        self.__keras_network = Sequential(
            [Dense(size, activation=funcs[i]) for i, size in
             enumerate(layers)])
        self.__keras_network.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        self.batch_size = batch_size

    def predict(self, data: Data) -> Label:
        flatten = False
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            flatten = True
        res = self.__keras_network.predict(data)
        if flatten:
            res = res.flatten()
        return res

    def train(self, data: Data, label: Label):
        return self.__keras_network.fit(data, label, batch_size=self.batch_size, verbose=False)
