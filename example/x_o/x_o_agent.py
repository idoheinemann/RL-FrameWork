import numpy as np

from rl_base.agent.agent import Agent
from rl_base.ml.neural_network import NeuralNetwork
from rl_base.ml.tools.functions import Tanh


class XOAgent(Agent):

    def __init__(self, name, gamma=0.9, epsilon=0.1):
        super().__init__()
        self.name = name
        self.gamma = gamma
        self.epsilon = epsilon
        self.net = NeuralNetwork((3 * 3 * 3, 4 * 4 * 4, 3 * 3), Tanh, alpha=0.01, n_min=-0.1, n_max=0.1)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand(9)
        return self.net.predict(state)

    def conclude(self):
        data, label = [], []
        value = 0
        for state, action, reward in reversed(self.memory):
            value = value * self.gamma + reward.get()
            label_temp = self.net.predict(state)
            label_temp[action] = value
            data.append(state)
            label.append(label_temp)

        self.net.train(np.array(data), np.array(label))
        self.memory.clear()

    def __str__(self):
        return f'XOAgent({self.name})'
