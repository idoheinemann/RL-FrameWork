import numpy as np

from example.snake.snake_environment import SnakeEnvironment
from rl_base.agent.agent import Agent
from rl_base.ml.neural_network import NeuralNetwork
from rl_base.ml.tools.functions import Tanh, Linear, Relu, SoftMax, LinearRelu, Sigmoid


class SnakeAgent(Agent):
    def choose_action(self, state):
        if self.epsilon > 0.01:
            self.epsilon *= 0.999
        else:
            self.epsilon = 0.01
        if np.random.rand() < self.epsilon and self.allow_random_actions:
            return np.random.rand(3)
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

    def __init__(self):
        super().__init__()
        self.model = NeuralNetwork((11, 64, 64, 3), (Sigmoid, Sigmoid, SoftMax), alpha=0.1,
                                   n_min=-0.5, n_max=0.5, gradient_method=None)#np.vectorize(lambda x: max(min(x, 10), -10)))
        self.epsilon = 1.0
        self.gamma = 0.9
        self.max_memory_size = 500
        self.memory_state = []
        self.losses = []
        self.rewards = []
        self.allow_random_actions = True
