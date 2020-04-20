import numpy as np

from rl_base.agent.agent import Agent
from rl_base.ml.recurrent_neural_network import RecurrentNeuralNetwork
from rl_base.ml.tools.functions import Relu, Sigmoid


class PrisonerDilemmaAgent(Agent):
    def __init__(self, name, gamma=0.99, epsilon=0.1):
        super().__init__()
        self.name = name
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = RecurrentNeuralNetwork((2, 8, 2), Sigmoid, max_unfold=5, n_min=-0.01, n_max=0.01, alpha=0.000001,
                                            use_saved_state_in_training=True)
        self.count = 0
        self.cache = []

    def choose_action(self, state):
        r = np.random.rand()
        if r < self.epsilon:
            return np.random.rand(2)
        return self.model.predict(state)

    def conclude(self):
        self.count += 1
        self.cache.append(self.memory)
        self.memory = []
        if self.count % 20 == 0:
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
                    label.append(label_temp)
                big_data.append(data)
                big_label.append(label)

            self.model.train(big_data, big_label)
            self.cache.clear()