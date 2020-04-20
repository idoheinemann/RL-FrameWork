from typing import Iterable

from rl_base.ml.prediction_model import PredictionModel, Data, Label
import numpy as np

from rl_base.ml.tools.functions import Sigmoid


class RecurrentNeuralNetwork(PredictionModel):
    def __init__(self, layers_num: Iterable[int], non_linear, n_min=-1.0, n_max=1.0, alpha=0.01, max_unfold=5,
                 use_saved_state_in_training=False):
        layers_num = list(layers_num)
        layers_size = [(layers_num[0], layers_num[1])]
        for i in range(1, len(layers_num) - 1):
            layers_size.append((layers_num[i], layers_num[i + 1]))
        self.W = []
        self.U = []
        self.B = []
        self.w_derivatives = None
        self.u_derivatives = None
        self.b_derivatives = None
        self.alpha = alpha
        self.max_unfold = max_unfold
        self.use_saved_state_in_training = use_saved_state_in_training
        if hasattr(non_linear, '__iter__') or hasattr(non_linear, '__getitem__'):
            assert len(layers_size) == len(non_linear)
        else:
            non_linear = (non_linear,) * len(layers_size)
        self.functions = non_linear
        for i in range(len(layers_size)):
            self.U.append((n_max - n_min) * np.random.rand(layers_size[i][0], layers_size[i][1]) + n_min)
            self.W.append((n_max - n_min) * np.random.rand(layers_size[i][1], layers_size[i][1]) + n_min)
            self.B.append((n_max - n_min) * np.random.rand(layers_size[i][1]) + n_min)
        self.states = []
        self.reset()

    def reset(self):
        self.states = [[np.array([0])]]

        for i in self.W:
            self.states[0].append(np.array([np.zeros(i.shape[0])]))

    def predict(self, data: Data) -> Label:
        prev_state = self.states[-1]
        state = [data]

        for j in range(len(self.U)):
            data = self.functions[j].calc(data.dot(self.U[j]) + prev_state[j + 1].dot(self.W[j]) + self.B[j])
            state.append(data)
        self.states.append(state)
        return state[-1].copy()

    def __unfold(self, layer, delta, states, step):
        if step <= 0:
            return
        state = states[-1]
        prev_state = states[-2]
        for j in range(layer, -1, -1):
            self.u_derivatives[j] += state[j].T.dot(delta)
            self.w_derivatives[j] += prev_state[j + 1].T.dot(delta)
            self.b_derivatives[j] += delta.reshape(self.b_derivatives[j].shape)
            delta_t = delta.dot(self.W[j].T) * self.functions[j].deriv(prev_state[j + 1])
            self.__unfold(layer=j, delta=delta_t, states=states[:-1], step=min(step - 1, len(states) - 1))
            if j > 0:
                delta = delta.dot(self.U[j].T) * self.functions[j - 1].deriv(state[j])

    def __get_gradient(self, labels):
        self.u_derivatives = []
        self.w_derivatives = []
        self.b_derivatives = []
        for i in range(len(self.U)):
            self.u_derivatives.append(np.zeros_like(self.U[i]))
            self.w_derivatives.append(np.zeros_like(self.W[i]))
            self.b_derivatives.append(np.zeros_like(self.B[i]))

        for i in range(len(labels)):
            state = self.states[i + 1]
            output = state[-1]
            prev_state = self.states[i]
            error = output - labels[i]
            delta = error * self.functions[-1].deriv(output)
            for j in range(len(self.U) - 1, -1, -1):
                self.u_derivatives[j] += state[j].T.dot(delta)
                self.w_derivatives[j] += prev_state[j + 1].T.dot(delta)
                self.b_derivatives[j] += delta.reshape(self.b_derivatives[j].shape)
                delta_t = delta.dot(self.W[j].T) * self.functions[j].deriv(prev_state[j + 1])
                self.__unfold(layer=j, delta=delta_t, states=self.states[:i + 1], step=min(self.max_unfold, i))
                if j > 0:
                    delta = delta.dot(self.U[j].T) * self.functions[j - 1].deriv(state[j])

    def train(self, data: Data, label: Label):
        gu = []
        for i in self.U:
            gu.append(np.zeros_like(i))
        gw = []
        for i in self.W:
            gw.append(np.zeros_like(i))
        gb = []
        for i in self.B:
            gb.append(np.zeros_like(i))
        for i in range(len(data)):
            if not self.use_saved_state_in_training:
                self.reset()
                for j in data[i]:
                    self.predict(j)
            self.__get_gradient(label[i])
            for j in range(len(gu)):
                gu[j] += self.u_derivatives[j]
                gw[j] += self.w_derivatives[j]
                gb[j] += self.b_derivatives[j]

        for i in range(len(self.U)):
            self.U[i] -= self.alpha * gu[i]
            self.W[i] -= self.alpha * gw[i]
            self.B[i] -= self.alpha * gb[i]


def main():
    rnn = RecurrentNeuralNetwork((2, 3, 1), Sigmoid)
    data, label = [], []
    for i in range(10):
        tmp_data = []
        carry = 0
        tmp_label = []
        for j in range(np.random.randint(5, 11)):
            tmp_data.append([[np.random.randint(0, 2), np.random.randint(0, 2)]])
            s = sum(tmp_data[-1][0]) + carry
            tmp_label.append([[s % 2]])
            carry = int(s > 1)
        data.append(np.array(tmp_data))
        label.append(np.array(tmp_label))

    rnn.reset()
    for i, d in enumerate(data[0]):
        print(rnn.predict(d), label[0][i], sep=' | ')

    for i in range(100000):
        rnn.train(data, label)

    print('Finished Training')

    rnn.reset()
    for i, d in enumerate(data[0]):
        print(rnn.predict(d), label[0][i], sep=' | ')


if __name__ == '__main__':
    main()
