import abc

import numpy as np

from rl_base.agent.agent import Agent
from rl_base.ml.prediction_model import PredictionModel


class BoardGameAgent(Agent, abc.ABC):
    def __init__(self, model: PredictionModel, actions_amount, gamma=0.9, epsilon=0.1):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = model
        self.actions_amount = actions_amount

    def choose_action(self, state):
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
            data.append(state)
            label.append(label_temp)

        self.model.train(np.array(data), np.array(label))
        self.memory.clear()
