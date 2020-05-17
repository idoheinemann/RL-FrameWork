import numpy as np

from rl_base.agent.agent import Agent
from rl_base.ml.recurrent_neural_network import RecurrentNeuralNetwork
from rl_base.ml.tools.functions import Relu, Sigmoid, SoftMax


class PrisonerDilemmaAgent(Agent):
    def __init__(self, name, gamma=0.5, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
        super().__init__()
        self.name = name
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.model = RecurrentNeuralNetwork((2, 16, 2), (Sigmoid, SoftMax), max_unfold=3, n_min=-0.5, n_max=0.1,
                                            alpha=0.5,
                                            use_saved_state_in_training=False)
        self.count = 0
        self.cache = []

    def choose_action(self, state):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        if np.random.rand() < self.epsilon:
            return np.random.rand(2)
        return self.model.predict(state)

    def conclude(self):
        self.cache.append(self.memory)
        self.memory = []
        if len(self.cache) > 100:
            self.cache.pop(0)

        big_label = []
        big_data = []
        for memory in self.cache:
            data, label = [], []
            value = 0
            self.model.reset()
            for index, (state, action, reward) in enumerate(reversed(memory)):
                value = value * self.gamma + reward.get()
                label_temp = self.model.predict(state)
                label_temp[0][action] = value
                data.append(state)
                label.append(label_temp / label_temp.sum())
            big_data.append(data)
            big_label.append(label)

        self.model.train(big_data, big_label)
