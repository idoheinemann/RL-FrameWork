from rl_base.agent.board_game_agent import BoardGameAgent
from rl_base.ml.neural_network import NeuralNetwork
from rl_base.ml.tools.functions import Tanh, Relu, Linear
import numpy as np


def normalize(x):
    return max(min(1, x), -1)


class XOAgent(BoardGameAgent):

    def __init__(self, name):
        super().__init__(
            NeuralNetwork((3 * 3 * 3, 64, 3 * 3), (Relu, Linear), alpha=0.01, n_min=-0.1, n_max=0.1,
                          gradient_method=np.vectorize(normalize)), 9,
            gamma=0.9, epsilon_decay=0.9999)
        self.name = name

    def __str__(self):
        return f'XOAgent({self.name})'
