import abc

import numpy as np

from rl_base.agent.agent import Agent
from rl_base.ml.prediction_model import PredictionModel


class BoardGameAgent(Agent, abc.ABC):
    def __init__(self, model: PredictionModel, actions_amount, gamma=0.9, epsilon=1.0, epsilon_decay=0.99,
                 min_epsilon=0.01, cache_size=100, normalize_reward=False):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.model = model
        self.actions_amount = actions_amount
        self.cache_data = []
        self.cache_label = []
        self.cache_size = cache_size
        self.normalize_reward = normalize_reward

    def choose_action(self, state):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.actions_amount)
        return self.model.predict(state)

    def conclude(self):
        data, label = [], []
        value = 0
        for state, action, reward in reversed(self.memory):
            value = value * self.gamma + reward.get()
            label_temp = self.model.predict(state)
            label_temp[action] = value
            if self.normalize_reward:
                label_temp -= label_temp.min()
                label_temp /= label_temp.sum()
            data.append(state)
            label.append(label_temp)

        self.cache_data += data
        self.cache_label += label
        if len(self.cache_data) > self.cache_size:
            self.cache_data = self.cache_data[len(self.cache_data) - self.cache_size:]
            self.cache_label = self.cache_label[len(self.cache_label) - self.cache_size:]

        self.model.train(np.array(self.cache_data), np.array(self.cache_label))
        self.memory = []
