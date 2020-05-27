import abc

import numpy as np

from rl_base.agent.agent import Agent
from rl_base.ml.prediction_model import PredictionModel


class VideoGameAgent(Agent, abc.ABC):
    def choose_action(self, state):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon
        if np.random.rand() < self.epsilon and self.allow_random_actions:
            return np.random.rand(self.flat_action_space)
        return self.model.predict(state).flatten()

    def learn(self, state, action, reward, new_state=None):
        self.rewards.append(reward)
        self.memory_state.append((state, new_state, reward, action))
        if len(self.memory_state) > self.max_memory_size:
            self.memory_state.pop(0)
        data = []
        label = []
        for state, new_state, reward, action in self.memory_state:
            data.append(state)
            _reward = self.model.predict(state)
            if reward < 0:
                _reward[action] = reward
            else:
                _reward[action] = reward + self.gamma * (np.amax(self.model.predict(new_state)))
            _reward -= _reward.min()
            _reward /= _reward.sum()
            label.append(_reward)
        self.losses.append(self.model.train(np.array(data), np.array(label)))

    def conclude(self):
        pass

    def __init__(self, model: PredictionModel, flat_action_space: int, epsilon=1.0, gamma=0.9, max_memory_size=500, allow_random_actions=True,
                 min_epsilon=0.01, epsilon_decay=0.99):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.flat_action_space = flat_action_space
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_memory_size = max_memory_size
        self.memory_state = []
        self.losses = []
        self.rewards = []
        self.allow_random_actions = allow_random_actions
